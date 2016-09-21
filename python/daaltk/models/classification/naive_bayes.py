"""
Intel DAAL Naive Bayes Model
"""

from sparktk.frame.frame import Frame
from sparktk.propobj import PropertiesObject
from sparktk.frame.ops.classification_metrics_value import ClassificationMetricsValue
from sparktk import TkContext
from daaltk import Daal

__all__ = ['train','load','NaiveBayesModel']

def train(frame, label_column, observation_columns, num_classes = 2):
    """
    Build a multinomial DAAL naive bayes model.

    Parameters
    ----------

    :param frame: (Frame) A frame to train the model on.
    :param label_column: (str) Column containing the label for each observation.
    :param observation_columns: (List[str]) Column(s) containing the observations
    :param num_classes: (int) Number of classes
    :return: (NaiveBayesModel) Trained Naive Bayes model

    """
    if not isinstance(frame, Frame):
        raise TypeError("frame parameter must be a sparktk frame, but received: %s" % type(frame))
    if not isinstance(label_column, basestring):
        raise TypeError("label_column must be a string, but received: %s" % type(label_column))
    if not isinstance(observation_columns, list) or not all(isinstance(column, basestring) for column in observation_columns):
        raise TypeError("observation_columns must be a list of strings")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an int, but received: %s" % type(num_classes))

    tc = frame._tc
    _scala_obj = _get_scala_obj(tc)
    scala_columns = tc.jutils.convert.to_scala_vector_string(observation_columns)
    scala_model = _scala_obj.train(frame._scala, label_column, scala_columns, num_classes)

    return NaiveBayesModel(tc, scala_model)

def _get_scala_obj(tc):
    """Gets reference to the scala object"""
    return tc.sc._jvm.org.trustedanalytics.daaltk.models.classification.naive_bayes.NaiveBayesModel

def load(path, tc=TkContext.implicit):
    """
    Load a DAAL NaiveBayesModel object from the given path.  An error is raised if the object provided is not a
    DAAL NaiveBayesModel.

    :param path: (str) Path to load
    :param tc: (TkContext) spark-tk context
    :return: (NaiveBayesModel) DAAL NaiveBayesModel loaded from the specified path
    """
    if isinstance(tc, Daal):
        tc = tc._tc
    return tc.load(path, NaiveBayesModel)


class NaiveBayesModel(PropertiesObject):
    """
    A trained Intel DAAL NaiveBayesModel.

    Naive Bayes [1]_ is a probabilistic classifier with strong independence assumptions between features. It computes
    the conditional probability distribution of each feature given label, and then applies Bayes' theorem to compute
    the conditional probability distribution of a label given an observation, and use it for prediction.

    The Naive Bayes model is initialized, trained on columns of a frame, tested against true labels of a frame and used
    to predict the value of the dependent variable given the independent observations of a frame and test the
    performance of the classification on test data. This model runs the Intel DAAL implementation of Naive Bayes [2]_.

    .. rubric:: footnotes

    .. [1] https://en.wikipedia.org/wiki/Naive_Bayes_classifier
    .. [2] https://software.intel.com/en-us/daal

    Example
    -------

        <hide>
        >>> data = [[1,19.8446136104,2.2985856384],[1,16.8973559126,2.6933495054],
        ...    [1,5.5548729596,2.7777687995],[0,46.1810010826,3.1611961917],
        ...    [0,44.3117586448,3.3458963222],[0,34.6334526911,3.6429838715]]
        >>> schema = [('Class', int), ('Dim_1', float), ('Dim_2', float)]
        >>> frame = tc.frame.create(data, schema)
        <progress>
        </hide>

    Consider the following frame containing three columns, which will be used to train and test the model.

        >>> frame.inspect()
        [#]  Class  Dim_1          Dim_2
        =======================================
        [0]      1  19.8446136104  2.2985856384
        [1]      1  16.8973559126  2.6933495054
        [2]      1   5.5548729596  2.7777687995
        [3]      0  46.1810010826  3.1611961917
        [4]      0  44.3117586448  3.3458963222
        [5]      0  34.6334526911  3.6429838715

    Train the model using the sample dataset in the frame:

        >>> model = tc.daaltk.models.classification.naive_bayes.train(frame, 'Class', ['Dim_1','Dim_2'], num_classes=2)

        >>> model.class_log_prior
        [-0.6931471805599453, -0.6931471805599453]

        >>> model.feature_log_prob
        [[-0.07696104113612832, -2.6026896854443837],
        [-0.15762894420358317, -1.9252908618525777]]

    Predict using the trained model:

        >>> predicted_frame = model.predict(frame)

        >>> predicted_frame.inspect()
        [#]  Class  Dim_1          Dim_2         predicted_class
        ========================================================
        [0]      1  19.8446136104  2.2985856384              0.0
        [1]      1  16.8973559126  2.6933495054              1.0
        [2]      1   5.5548729596  2.7777687995              1.0
        [3]      0  46.1810010826  3.1611961917              0.0
        [4]      0  44.3117586448  3.3458963222              0.0
        [5]      0  34.6334526911  3.6429838715              0.0

        >>> test_metrics = model.test(frame, 'Class')
        <progress>

        >>> test_metrics
        accuracy         = 0.833333333333
        confusion_matrix =             Predicted_Pos  Predicted_Neg
        Actual_Pos              2              1
        Actual_Neg              0              3
        f_measure        = 0.8
        precision        = 1.0
        recall           = 0.666666666667

    """

    def __init__(self, tc, scala_model):
        self._tc = tc
        tc.jutils.validate_is_jvm_instance_of(scala_model, _get_scala_obj(tc))
        self._scala = scala_model

    @staticmethod
    def _from_scala(tc, scala_model):
        return NaiveBayesModel(tc, scala_model)

    @property
    def label_column(self):
        """
        Column name containing the label
        """
        return self._scala.labelColumn()

    @property
    def observation_columns(self):
        """
        List of column(s) storing the observations
        """
        return self._tc.jutils.convert.from_scala_seq(self._scala.trainingObservationColumns())

    @property
    def lambda_parameter(self):
        """
        Additive smoothing parameter
        """
        return self._scala.lambdaParameter()

    @property
    def class_log_prior(self):
        """
        Smoothed empirical log probability for each class.
        """
        return list(self._scala.classLogPrior())

    @property
    def feature_log_prob(self):
        """
        Empirical log probability of features given a class, P(x_i|y).
        """
        return [list(i) for i in list(self._scala.featureLogProb())]

    def predict(self, frame, columns=None):
        """
        Predict labels for data points using trained multinomial Naive Bayes model.

        Parameters
        ----------

        :param frame: (Frame) A frame whose labels are to be predicted.
        :param columns: (Optional(list[str])) Column(s) containing the observations whose labels are to be predicted.
                        By default, we predict the labels over columns the NaiveBayesModel was trained on.
        :return: (Frame) Frame containing the original frame's columns and a column with the predicted label.
        """
        if not isinstance(frame, Frame):
            raise TypeError("frame parameter must be a sparktk frame, but received: %s" % type(frame))
        if columns is not None and not isinstance(columns, list):
            raise TypeError("columns parameter must be a list of strings (or None), but received %s" % type(columns))
        scala_columns = self._tc.jutils.convert.to_scala_option_list_string(columns)
        return Frame(self._tc, self._scala.predict(frame._scala, scala_columns))

    def test(self, frame, label_column, columns=None):
        """
        Predict test frame labels and return metrics.  Run classification metrics on predicted and target labels.

        Parameters
        ----------

        :param frame: (Frame) A frame whose labels are to be predicted.
        :param label_column: (str) Column containing the actual label for each observation.
        :param columns: (Optional(list[str])) Column(s) containing the observations whose labels are to be predicted.
                        By default, we predict the labels over the columns the NaiveBayesModel was trained on.
        :return: (ClassificationMetricsValue) Object with classification metrics

        """
        if not isinstance(frame, Frame):
            raise TypeError("frame parameter must be a sparktk frame, but received: %s" % type(frame))
        if not isinstance(label_column, basestring):
            raise TypeError("label_column parameter must be a str, but received: %s" % type(label_column))
        if columns is not None and not isinstance(columns, list):
            raise TypeError("columns parameter must be a list of strings (or None), but received %s" % type(columns))

        scala_columns = self._tc.jutils.convert.to_scala_option_list_string(columns)
        return ClassificationMetricsValue(self._tc, self._scala.test(frame._scala, unicode(label_column), scala_columns))

    def save(self, path):
        """
        Save the trained model to the specified path
        :param path: Path to save
        """
        self._scala.save(self._tc._scala_sc, path)