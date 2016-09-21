"""
Intel DAAL Linear Regression Model
"""

from sparktk.frame.frame import Frame
from sparktk.propobj import PropertiesObject
from sparktk import TkContext
from daaltk import Daal

__all__ = ['train','load','LinearRegressionModel']

def train(frame, value_column, observation_columns, fit_intercept=True):
    """
    Build Intel DAAL linear regression model.

    Parameters
    ----------

    :param frame: (Frame) A frame to train or test the model on.
    :param value_column: (string) Column name containing the value for each observation.
    :param observation_columns: (List[string]) List of column(s) containing the observations
    :param fit_intercept: (Optional(bool)) Parameter for whether to fit an intercept term.
    :return: (LinearRegressionModel) Trained Intel DAAL linear regression model.
    """
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a sparktk Frame, but is: %s" % type(frame))
    if not isinstance(value_column, basestring):
        raise TypeError("value_column must be a string, but is: %s" % type(value_column))
    if not isinstance(observation_columns, list) or not all(isinstance(column, basestring) for column in observation_columns):
        raise TypeError("observation_columns must be a list of strings")
    if not isinstance(fit_intercept, bool):
        raise TypeError("fit_intercept must be a bool, but is: %s", type(fit_intercept))
    tc = frame._tc
    _scala_obj = _get_scala_obj(tc)
    scala_model = _scala_obj.train(frame._scala,
                                   value_column,
                                   tc.jutils.convert.to_scala_list_string(observation_columns),
                                   fit_intercept)
    return LinearRegressionModel(tc, scala_model)

def _get_scala_obj(tc):
    """Gets reference to the scala object"""
    return tc.sc._jvm.org.trustedanalytics.daaltk.models.regression.linear_regression.LinearRegressionModel

def load(path, tc=TkContext.implicit):
    """
    Load a DAAL LinearRegressionModel object from the given path.  An error is raised if the object provided is not a
    DAAL LinearRegressionModel.

    :param path: (str) Path to load
    :param tc: (TkContext) spark-tk context
    :return: (LinearRegressionModel) DAAL LinearRegressionModel loaded from the specified path
    """
    if isinstance(tc, Daal):
        tc = tc._tc
    return tc.load(path, LinearRegressionModel)

class LinearRegressionModel(PropertiesObject):
    """
    Linear Regression [1]_ is used to model the relationship between a scalar dependent variable and one or more
    independent variables.
    The Linear Regression model is initialized, trained on columns of a frame and used to predict the value of the
    dependent variable given the independent observations of a frame.
    This model runs the DAAL implementation of Linear Regression [2]_ with QR [3]_ decomposition.

    .. rubric:: footnotes

    .. [1] https://en.wikipedia.org/wiki/Linear_regression
    .. [2] https://software.intel.com/en-us/daal
    .. [3] https://en.wikipedia.org/wiki/QR_decomposition

    Example
    -------

        <hide>
        >>> data = [[0,0],[1, 2.5],[2, 5.0],[3, 7.5],[4, 10],[5, 12.5],[6, 13.0],[7, 17.15], [8, 18.5],[9, 23.5]]
        >>> schema = [("x1", float),("y", float)]
        >>> frame = tc.frame.create(data, schema)
        <progress>

        </hide>

    Consider the following model trained and tested on the sample data set in the frame, which has two columns:

        >>> frame.inspect()
        [#]  x1  y
        ==============
        [0]   0      0
        [1]   1    2.5
        [2]   2    5.0
        [3]   3    7.5
        [4]   4     10
        [5]   5   12.5
        [6]   6   13.0
        [7]   7  17.15
        [8]   8   18.5
        [9]   9   23.5

        >>> model = tc.daaltk.models.regression.linear_regression.train(frame,'y',['x1'])
        >>> model
        explained_variance      = 49.2759280303
        intercept               = -0.0327272727273
        mean_absolute_error     = 0.529939393939
        mean_squared_error      = 0.630096969697
        observation_columns     = [u'x1']
        r2                      = 0.987374330661
        root_mean_squared_error = 0.793786476136
        value_column            = y
        weights                 = [2.443939393939393]

        >>> test_output = model.test(frame,'y')
        >>> test_output
        explained_variance      = 49.2759280303
        mean_absolute_error     = 0.529939393939
        mean_squared_error      = 0.630096969697
        r2                      = 0.987374330661
        root_mean_squared_error = 0.793786476136

        >>> predicted_frame = model.predict(frame, observation_columns = ["x1"])
        >>> predicted_frame.inspect()
        [#]  x1   y      predict_y
        =================================
        [0]  0.0    0.0  -0.0327272727273
        [1]  1.0    2.5     2.41121212121
        [2]  2.0    5.0     4.85515151515
        [3]  3.0    7.5     7.29909090909
        [4]  4.0   10.0     9.74303030303
        [5]  5.0   12.5      12.186969697
        [6]  6.0   13.0     14.6309090909
        [7]  7.0  17.15     17.0748484848
        [8]  8.0   18.5     19.5187878788
        [9]  9.0   23.5     21.9627272727

    """
    def __init__(self, tc, scala_model):
        self._tc = tc
        tc.jutils.validate_is_jvm_instance_of(scala_model, _get_scala_obj(tc))
        self._scala = scala_model

    @staticmethod
    def _from_scala(tc, scala_model):
        return LinearRegressionModel(tc, scala_model)

    @property
    def observation_columns(self):
        """
        List of column(s) storing the observations
        """
        return list(self._scala.observationColumnsAsArray())

    @property
    def value_column(self):
        """
        Column name containing the value for each observation
        """
        return self._scala.valueColumnTrain()

    @property
    def intercept(self):
        """
        Intercept of the trained model
        """
        return self._scala.intercept()

    @property
    def weights(self):
        """
        Weights of the trained model
        """
        return list(self._scala.weightsAsArray())

    @property
    def explained_variance(self):
        """
        The explained variance regression score
        """
        return self._scala.explainedVariance()

    @property
    def mean_absolute_error(self):
        """
        The risk function corresponding to the expected value of the absolute error loss or l1-norm loss
        """
        return self._scala.meanAbsoluteError()

    @property
    def mean_squared_error(self):
        """
        The risk function corresponding to the expected value of the squared error loss or quadratic loss
        """
        return self._scala.meanSquaredError()

    @property
    def r2(self):
        """
        The coefficient of determination of the trained model
        """
        return self._scala.r2()

    @property
    def root_mean_squared_error(self):
        """
        The square root of the mean squared error
        """
        return self._scala.rootMeanSquaredError()

    def test(self, frame, value_column=None, observation_columns=None):
        """
        Compute test metrics for trained Intel DAAL linear regression model.

        Parameters
        ----------

        :param frame: (Frame) The frame to test the linear regression model on
        :param value_column: (Optional(string)) Column name containing the value of each observation
        :param observation_columns: (Optional(list[string])) List of column(s) containing the observations.
        :return: (LinearRegressionTestReturn) Test metrics for Intel DAAL linear regression model
        """
        if not isinstance(frame, Frame):
            raise TypeError("frame parameter should be a frame, but received type: %s" % type(frame))
        if value_column is not None and not isinstance(value_column, basestring):
            raise TypeError("value_column should be a string (or None), but received type: %s" % type(value_column))
        if observation_columns is not None and not isinstance(observation_columns, list):
            raise TypeError("observation_columns must be a list of strings (or None).")

        scala_result = self._scala.test(frame._scala,
                                        self._tc.jutils.convert.to_scala_option(value_column),
                                        self._tc.jutils.convert.to_scala_option_list_string(observation_columns))
        return LinearRegressionTestReturn(scala_result)

    def predict(self, frame, observation_columns=None, label_column=None):
        """
        Predict the cluster assignments for the data points, using the trained model.

        Parameters
        ----------

        :param frame: (Frame)A frame whose labels are to be predicted.
        :param observation_columns: (Optional(list[str])) Column(s) containing the observations whose clusters are to be
                                    predicted.  Default is to predict the clusters over the columns that the KMeans model
                                    was trained on.
        :param label_column: (Optional(str)) Name of the output column with index of cluster each observation belongs to.
        :return: (Frame) A new frame consisting of the existing columns of the frame and the following new columns 'k'
                 columns: Each of the 'k' columns containing squared distance of that observation to the 'k'th cluster
                 center predicted_cluster column: The cluster assignment for the observation.

        """
        if not isinstance(frame, Frame):
            raise TypeError("'frame' parameter should be a frame, but received type: %s." % type(frame))
        if observation_columns != None and (not isinstance(observation_columns, list) or not all(isinstance(column, basestring) for column in observation_columns)):
            raise TypeError("observation_columns must be a list of strings (or None)")
        if label_column != None and not isinstance(label_column, basestring):
            raise TypeError("label_column must be a string (or None)")

        predict_frame = self._scala.predict(frame._scala,
                                            self._tc.jutils.convert.to_scala_option(label_column),
                                            self._tc.jutils.convert.to_scala_option_list_string(observation_columns))

        return Frame(self._tc, predict_frame)

    def save(self, path):
        """
        Save the trained model to the specified path
        :param path: Path to save
        """
        self._scala.save(self._tc._scala_sc, path)


class LinearRegressionTestReturn(PropertiesObject):
    """
    LinearRegressionTestReturn class contains values that are returned from the Intel DAAL LinearRegressionModel test()
    """
    def __init__(self, scala_result):
        self._explained_variance = scala_result.explainedVariance()
        self._mean_absolute_error = scala_result.meanAbsoluteError()
        self._mean_squared_error = scala_result.meanSquaredError()
        self._r2 = scala_result.r2()
        self._root_mean_squared_error = scala_result.rootMeanSquaredError()

    @property
    def explained_variance(self):
        """
        The explained variance regression score
        """
        return self._explained_variance

    @property
    def mean_absolute_error(self):
        """
        The risk function corresponding to the expected value of the absolute error loss or l1-norm loss
        """
        return self._mean_absolute_error

    @property
    def mean_squared_error(self):
        """
        The risk function corresponding to the expected value of the squared error loss or quadratic loss
        """
        return self._mean_squared_error

    @property
    def r2(self):
        """
        The coefficient of determinatio
        """
        return self._r2

    @property
    def root_mean_squared_error(self):
        """
        The square root of the mean squared error
        """
        return self._root_mean_squared_error