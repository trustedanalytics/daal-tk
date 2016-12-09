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
Intel DAAL KMean Model
"""

from sparktk.frame.frame import Frame
from sparktk.propobj import PropertiesObject
from sparktk import TkContext
from daaltk import Daal
from sparktk.arguments import require_type

__all__ = ['train','load','KMeansModel']

def train(frame, observation_columns, column_scalings=None, k=2, max_iterations=100, label_column='predicted_cluster'):
    """
    Creates DAAL KMeans Model using the observation columns.  The algorithm chooses random observations as the
    initial cluster centers.

    Parameters
    ----------

    :param frame: (Frame) A frame to train the model on
    :param observation_columns: (list[string]) Columns containing the observations.
    :param column_scalings: (Optional(list[float])) Optional column scalings for each of the observation columns. The
                            scaling value is multiplied by the corresponding value in the observation column.
    :param k: (Optional(int)) Desired number of clusters.
    :param max_iterations: (Optional(int)) Number of iterations for which the algorithm should run.
    :param label_column: (Optional(string)) Optional name of the output column with index of cluster each observation
                         belongs to
    :return: (KMeansModel) Trained Intel DAAL KMeans Model
    """
    if not isinstance(frame, Frame):
        raise TypeError("frame parameter must be a sparktk frame, but received: %s" % type(frame))
    if not isinstance(observation_columns, list) or \
            not all(isinstance(column, basestring) for column in observation_columns):
        raise TypeError("observation_columns must be a list of strings")
    if not isinstance(k, int):
        raise TypeError("k must be an int, but is: %s" % type(k))
    if not isinstance(max_iterations, int):
        raise TypeError("max_iterations must be an int, but is: %s" % type(max_iterations))
    if not isinstance(label_column, basestring):
        raise TypeError("label_column must be a string, but is: %s" % type(label_column))
    if column_scalings != None and not isinstance(column_scalings, list):
        raise TypeError("column_scalings must be a list (or None), but is %s." % type(column_scalings))

    tc = frame._tc
    _scala_obj = _get_scala_obj(tc)

    # Train model
    scala_model = _scala_obj.train(frame._scala,
                                   tc.jutils.convert.to_scala_list_string(observation_columns),
                                   tc.jutils.convert.to_scala_option_list_double(column_scalings),
                                   k,
                                   max_iterations,
                                   label_column)

    # Return trained model
    return KMeansModel(tc, scala_model)

def load(path, tc=TkContext.implicit):
    """
    Load a DAAL KMeansModel object from the given path.  An error is raised if the object provided is not a DAAL KMeansModel.

    :param path: (str) Path to load
    :param tc: (TkContext) spark-tk context
    :return: (KMeansModel) DAAL KMeansModel loaded from the specified path
    """
    if isinstance(tc, Daal):
        tc = tc._tc
    return tc.load(path, KMeansModel)

def _get_scala_obj(tc):
    """Gets reference to the scala object"""
    return tc.sc._jvm.org.trustedanalytics.daaltk.models.clustering.kmeans.KMeansModel

class KMeansModel(PropertiesObject):
    """
    Trained Intel DAAL KMeans Model

    k-means [1]_ is an unsupervised algorithm used to partition
    the data into 'k' clusters.
    Each observation can belong to only one cluster, the cluster with the nearest
    mean.
    The k-means model is initialized, trained on columns of a frame, and used to
    predict cluster assignments for a frame.

    This model runs the DAAL implementation of k-means[2]_. The K-Means clustering
    algorithm computes centroids using the Lloyd method[3]_

    .. rubric:: footnotes

    .. [1] https://en.wikipedia.org/wiki/K-means_clustering
    .. [2] https://software.intel.com/en-us/daal
    .. [3] https://en.wikipedia.org/wiki/Lloyd%27s_algorithm

    Example
    -------

        <hide>
        >>> data = [[2,"ab"],[1,"cd"],[7,"ef"],[1,"gh"],[9,"ij"],[2,"kl"],[0,"mn"],[6,"op"],[5,"qr"], [120, "outlier"]]
        >>> schema = [("data", float),("name", str)]
        >>> frame = tc.frame.create(data, schema)
        <progress>

        </hide>

    Consider the following model trained and tested on the sample data set in the frame.  The frame has two columns:

        >>> frame.inspect()
        [#]  data  name
        ==================
        [0]     2  ab
        [1]     1  cd
        [2]     7  ef
        [3]     1  gh
        [4]     9  ij
        [5]     2  kl
        [6]     0  mn
        [7]     6  op
        [8]     5  qr
        [9]   120  outlier

    Train the model and view the model properties (including cluster centroids and sizes):

        >>> model = tc.daaltk.models.clustering.kmeans.train(frame, ["data"], k=2, max_iterations=20)

        >>> model
        centroids           = {u'Cluster:1': [120.0], u'Cluster:0': [3.6666666666666665]}
        cluster_sizes       = {u'Cluster:1': 1L, u'Cluster:0': 9L}
        column_scalings     = []
        k                   = 2
        label_column        = predicted_cluster
        observation_columns = [u'data']

    Call model to predict and then view the predicted_frame that is returned:

        >>> predicted_frame = model.predict(frame, ["data"])

        >>> predicted_frame.inspect()
        [#]  data   name     distance_from_cluster_0  distance_from_cluster_1
        =====================================================================
        [0]    2.0  ab                 2.77777777778                  13924.0
        [1]    1.0  cd                 7.11111111111                  14161.0
        [2]    7.0  ef                 11.1111111111                  12769.0
        [3]    1.0  gh                 7.11111111111                  14161.0
        [4]    9.0  ij                 28.4444444444                  12321.0
        [5]    2.0  kl                 2.77777777778                  13924.0
        [6]    0.0  mn                 13.4444444444                  14400.0
        [7]    6.0  op                 5.44444444444                  12996.0
        [8]    5.0  qr                 1.77777777778                  13225.0
        [9]  120.0  outlier            13533.4444444                      0.0
        <BLANKLINE>
        [#]  predicted_cluster
        ======================
        [0]                  0
        [1]                  0
        [2]                  0
        [3]                  0
        [4]                  0
        [5]                  0
        [6]                  0
        [7]                  0
        [8]                  0
        [9]                  1

    The trained model can be saved and restored:

        >>> model.save("sandbox/daal_kmeans")

        >>> restored_model = tc.load("sandbox/daal_kmeans")
        >>> predict_from_restored = restored_model.predict(frame, ["data"])
        >>> predict_from_restored.inspect()
        [#]  data   name     distance_from_cluster_0  distance_from_cluster_1
        =====================================================================
        [0]    2.0  ab                 2.77777777778                  13924.0
        [1]    1.0  cd                 7.11111111111                  14161.0
        [2]    7.0  ef                 11.1111111111                  12769.0
        [3]    1.0  gh                 7.11111111111                  14161.0
        [4]    9.0  ij                 28.4444444444                  12321.0
        [5]    2.0  kl                 2.77777777778                  13924.0
        [6]    0.0  mn                 13.4444444444                  14400.0
        [7]    6.0  op                 5.44444444444                  12996.0
        [8]    5.0  qr                 1.77777777778                  13225.0
        [9]  120.0  outlier            13533.4444444                      0.0
        <BLANKLINE>
        [#]  predicted_cluster
        ======================
        [0]                  0
        [1]                  0
        [2]                  0
        [3]                  0
        [4]                  0
        [5]                  0
        [6]                  0
        [7]                  0
        [8]                  0
        [9]                  1

    The trained model can also be exported to a .mar file, to be used with the scoring engine:

        >>> canonical_path = model.export_to_mar("sandbox/daalKMeans.mar")

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
        return KMeansModel(tc, scala_model)

    @property
    def observation_columns(self):
        """
        List of columns containing observations
        """
        return list(self._scala.observationColumnsAsArray())

    @property
    def k(self):
        """
        Number of clusters
        """
        return self._scala.k()

    @property
    def column_scalings(self):
        """
        Column scalings for each of the observation columns
        """
        return list(self._scala.columnScalingsAsArray())

    @property
    def label_column(self):
        """
        Column with index of cluster each observation belongs t
        """
        return self._scala.trainingLabelColumn()

    @property
    def centroids(self):
        """
        Dictionary of cluster names and centroids
        """
        centroids_dict = {}
        clusters = list(self._scala.clusters())
        centroids = list([list(c) for c in self._scala.centroids()])
        for cluster, centroid_list in zip(clusters, centroids):
            centroids_dict[cluster] = centroid_list
        return centroids_dict

    @property
    def cluster_sizes(self):
        """
        A dictionary of cluster names and sizes
        """
        clusters = list(self._scala.clusters())
        sizes = list(self._scala.clusterSize())
        cluster_dict = {}
        for cluster, size in zip(clusters, sizes):
            cluster_dict[cluster] = size
        return cluster_dict

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
                                            self._tc.jutils.convert.to_scala_option_list_string(observation_columns),
                                            self._tc.jutils.convert.to_scala_option(label_column))

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
