# vim: set encoding=utf-8

#  Copyright (c) 2016 Intel Corporation 
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
Intel DAAL Principal Components Model
"""

from sparktk.frame.frame import Frame
from sparktk.propobj import PropertiesObject
from sparktk import TkContext
from daaltk import Daal
from sparktk.arguments import require_type

__all__ = ['train','load','PrincipalComponentsModel']

def train(frame, columns, mean_centered=True, k=None):
    """
    Creates a DAAL PrincipalComponentsModel by training on the given frame

    Parameters
    ----------

    :param frame: (Frame) A frame to train the model on
    :param columns: (List[str]) List of column(s) containing the observations.
    :param mean_centered: (Optional(bool)) Option to mean center the columns.
    :param k: (Optional(int)) Principal component count.  Default is the number of observation columns.
    :return: (PrincipalComponentsModel) Trained PrincipalComponentsModel model
    """
    if not isinstance(frame, Frame):
        raise TypeError("frame parameter must be a sparktk frame, but received: %s" % type(frame))
    if not isinstance(columns, list) or not all(isinstance(column, basestring) for column in columns):
        raise TypeError("columns must be a list of strings")
    if mean_centered is not None and not isinstance(mean_centered, bool):
        raise TypeError("mean_centered must be a bool (or None), but received: %s" % type(mean_centered))
    if k is not None and not isinstance(k, int):
        raise TypeError("k must be an int (or None), but received: %s" % type(k))
    tc = frame._tc
    _scala_obj = _get_scala_obj(tc)
    scala_columns = tc.jutils.convert.to_scala_vector_string(columns)
    if not isinstance(mean_centered, bool):
        raise ValueError("mean_centered must be a bool, received %s" % type(mean_centered))
    scala_k = tc.jutils.convert.to_scala_option(k)
    scala_model = _scala_obj.train(frame._scala, scala_columns, mean_centered, scala_k)
    return PrincipalComponentsModel(tc, scala_model)

def _get_scala_obj(tc):
    """Gets reference to the scala object"""
    return tc.sc._jvm.org.trustedanalytics.daaltk.models.dimensionality_reduction.principal_components.PrincipalComponentsModel

def load(path, tc=TkContext.implicit):
    """
    Load a DAAL PrincipalComponentsModel object from the given path.  An error is raised if the object provided is not a
    DAAL PrincipalComponentsModel.

    :param path: (str) Path to load
    :param tc: (TkContext) spark-tk context
    :return: (PrincipalComponentsModel) DAAL PrincipalComponentsModel loaded from the specified path
    """
    if isinstance(tc, Daal):
        tc = tc._tc
    return tc.load(path, PrincipalComponentsModel)

class PrincipalComponentsModel(PropertiesObject):
    """
    Principal component analysis [1]_ is a statistical algorithm that converts possibly correlated features to
    linearly uncorrelated variables called principal components. The number of principal components is less than
    or equal to the number of original variables.

    This implementation of computing Principal Components is done by Singular Value Decomposition [2]_ of the data,
    providing the user with an option to mean center the data. The Principal Components model is initialized;
    trained on specifying the observation columns of the frame and the number of components; used to predict
    principal components.

    The Intel DAAL Singular Value Decomposition [3]_ implementation has been used for this, with additional features
    to 1) mean center the data during train and predict and 2) compute the t-squared index during prediction.

    .. rubric:: footnotes

    .. [1] https://en.wikipedia.org/wiki/Principal_component_analysis
    .. [2] https://en.wikipedia.org/wiki/Singular_value_decomposition
    .. [3] https://software.intel.com/en-us/daal

    Example
    -------

    In this example, the DAAL principal components model will be trained and tested on the sample dataset in the the
    frame.

        <hide>
        >>> data = [[2.6,1.7,0.3,1.5,0.8,0.7],
        ...        [3.3,1.8,0.4,0.7,0.9,0.8],
        ...        [3.5,1.7,0.3,1.7,0.6,0.4],
        ...        [3.7,1.0,0.5,1.2,0.6,0.3],
        ...        [1.5,1.2,0.5,1.4,0.6,0.4]]

        >>> schema = [("1", float),("2", float),("3", float),("4", float),("5", float),("6", float)]

        >>> frame = tc.frame.create(data, schema)
        <progress>

        </hide>

    Consider the following frame, which contains six columns:

        >>> frame.inspect()
        [#]  1    2    3    4    5    6
        =================================
        [0]  2.6  1.7  0.3  1.5  0.8  0.7
        [1]  3.3  1.8  0.4  0.7  0.9  0.8
        [2]  3.5  1.7  0.3  1.7  0.6  0.4
        [3]  3.7  1.0  0.5  1.2  0.6  0.3
        [4]  1.5  1.2  0.5  1.4  0.6  0.4

    Perform training, which returns a model:

        >>> model = tc.daaltk.models.dimensionality_reduction.principal_components.train(frame,['1','2','3','4','5','6'], mean_centered=True, k=3)
        -etc-

    Properties such as column_means, columns, and singlar_values can be accessed through the model object:

        <skip>
        >>> model
        column_means    = [2.92, 1.48, 0.4, 1.3, 0.7, 0.52]
        columns         = [u'1', u'2', u'3', u'4', u'5', u'6']
        k               = 3
        mean_centered   = True
        singular_values = [1.8056090475708324, 0.9698886054181584, 0.919656109986899]
        vfactor         = [[0.9895680076816261, 0.046382751353135805, -0.12475255750327569], [0.07897478506481442, -0.4943009041467917, 0.3176350198124238], [-0.03674645332401975, 0.7300190812545759, 0.0796350142543979], [-0.1121071574201765, -0.26091516308565543, -0.8523686966878925], [0.023546733633406085, -0.33883576500462137, 0.14001735739339258], [0.006736189980619886, -0.19416745238716243, 0.36203011287047626]]
        </skip>

    Predict using the same sample dataset that we used for training:

        >>> predicted_frame = model.predict(frame, mean_centered=True, t_squared_index=True, observation_columns=['1','2','3','4','5','6'], c=2)
        -etc-

    Take a look at the predicted_frame:

        <skip>
        >>> predicted_frame.inspect()
        [#]  1    2    3    4    5    6    p_1              p_2
        ===================================================================
        [0]  2.6  1.7  0.3  1.5  0.8  0.7  -0.314466908336  -0.317607338018
        [1]  3.3  1.8  0.4  0.7  0.9  0.8   0.475167548513  -0.106135785631
        [2]  3.5  1.7  0.3  1.7  0.6  0.4   0.546992663373    -0.2020285057
        [3]  3.7  1.0  0.5  1.2  0.6  0.3   0.737654584411   0.449136820506
        [4]  1.5  1.2  0.5  1.4  0.6  0.4   -1.44534788796   0.176634808843
        <BLANKLINE>
        [#]  t_squared_index
        ====================
        [0]    0.13756729239
        [1]  0.0812293200034
        [2]    0.13516232763
        [3]    0.38134474314
        [4]   0.673930077524
        </skip>

    The trained model can be saved and restored:

        >>> model.save("sandbox/daal_pca")

        >>> restored_model = tc.load("sandbox/daal_pca")
        >>> restored_model.columns
        [u'1', u'2', u'3', u'4', u'5', u'6']
    <skip>
        >>> restored_model.column_means
        [2.92, 1.48, 0.4, 1.3, 0.7, 0.52]
    </skip>
    <hide>
        >>> # check that column means are almost equal to the original model
        >>> for x, y in zip(restored_model.column_means, model.column_means):
        ...   assert(round(x, 3) == round(y, 3))
    </hide>
        >>> restored_model.k
        3

    The trained model can also be exported to a .mar file, to be used with the scoring engine:

        >>> canonical_path = model.export_to_mar("sandbox/daalPrincipalComponents.mar")

    <hide>
        >>> import os
        >>> assert(os.path.isfile(canonical_path))
    </hide>

    """

    def __init__(self, tc, scala_model):
        self._tc = tc
        tc.jutils.validate_is_jvm_instance_of(scala_model, _get_scala_obj(tc))
        self._scala = scala_model

    @staticmethod
    def _from_scala(tc, scala_model):
        return PrincipalComponentsModel(tc, scala_model)

    @property
    def columns(self):
        """
        List of observation column name(s) used to train the mode
        """
        return list(self._scala.observationColumns())

    @property
    def mean_centered(self):
        """
        Indicator whether the columns were mean centered for training
        """
        return self._scala.meanCentered()

    @property
    def k(self):
        """
        Principal component count
        """
        return self._scala.k()

    @property
    def column_means(self):
        """
        Means of the columns
        """
        return list(self._scala.columnMeans())

    @property
    def singular_values(self):
        """
        Singular values of the specified columns in the input frame
        """
        return list(self._scala.singularValues())

    @property
    def vfactor(self):
        """
        Right singular vectors of the specified columns in the input frame
        """
        return [list(i) for i in list(self._scala.vFactor())]

    def predict(self, frame, mean_centered=True, t_squared_index=False, observation_columns=None, c=None):
        """
        Predicting on a frame's columns using an Intel DAAL PrincipalComponentsModel.

        Parameters
        ----------

        :param frame: (Frame) Frame whose principal components are to be computed.
        :param mean_centered: (Optional(bool)) Option to mean center the columns.
        :param t_squared_index: (Optional(bool)) Indicator for whether the t-squared index is to be computed.
        :param observation_columns: (Option(list[str])) List of observation column name(s) to be used for prediction.
                                    Default is the list of column name(s) used to train the model.
        :param c: (Optional(int)) The number of principal components to be predicted. 'c' cannot be greater than the
                  count used to train the model.  Default is the count used to train the model.
        :return: (Frame) A frame with existing columns and following additional columns:
                 'c' additional columns: containing the projections of V on the the frame
                 't_squared_index': column storing the t-square-index value, if requested
        """
        if not isinstance(frame, Frame):
            raise TypeError("'frame' parameter should be a frame, but received type: %s." % type(Frame))
        if not isinstance(mean_centered, bool):
            raise TypeError("'mean_centered' parameter should be a boolean, but received type: %s." % type(mean_centered))
        if not isinstance(t_squared_index, bool):
            raise TypeError("'t_squared_index' parameter should be a boolean, but received type: %s." % type(t_squared_index))
        if c is not None and not isinstance(c, int):
            raise TypeError("If optional parameter 'c' is specified, it should be an int, but received type: %s." %(type(c)))
        if observation_columns is not None and not isinstance(observation_columns, list):
            raise TypeError("observation_columns must be a list of strings (or None), but received type: %s" % type(observation_columns))

        predict_frame = self._scala.predict(frame._scala,
                                            mean_centered,
                                            t_squared_index,
                                            self._tc.jutils.convert.to_scala_option_list_string(observation_columns),
                                            self._tc.jutils.convert.to_scala_option(c))

        return Frame(self._tc, predict_frame)

    def save(self, path):
        """
        Save the trained model to the specified path
        :param path: Path to save
        """
        self._scala.save(self._tc._scala_sc, path)

    def export_to_mar(self, path):
        """
        Exports the trained model as a model archive (.mar) to the specified path

        Parameters
        ----------

        :param path: (str) Path to save the trained model
        :return: (str) Full path to the saved .mar file
        """
        require_type.non_empty_str(path, "path")
        return self._scala.exportToMar(self._tc._scala_sc, path)