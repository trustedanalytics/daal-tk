#
# /*
# // Copyright (c) 2016 Intel Corporation 
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //      http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.
# */
#

from sparktk.frame.frame import Frame

def covariance_matrix(frame, data_column_names = None):
    """
    Uses the Intel Data Analytics and Acceleration Library (DAAL) to compute covariance matrix.

    Notes
    -----
    This function applies only to columns containing numerical data.

    Parameters
    ----------

    :param frame: (Frame) A frame of data to use for the operation.
    :param data_column_names: (Optional[List[str]]) The names of the column from which to compute the matrix. Names
                              should refer to a single column of type vector, or two or more columns of numeric scalars.
                              If no column names are provided, all columns in the frame will be used.
    :return: (Frame) A frame that contains a matrix with the covariance values for the columns.

    Example
    -------

        <hide>
        >>> schema = [("idnum", int), ("x1", float), ("x2", float), ("x3", float), ("x4", float)]
        >>> data = [[0, 1.0, 4.0, 0.0, -1.0],
        ...         [1, 2.0, 3.0, 0.0, -1.0],
        ...         [2, 3.0, 2.0, 1.0, -1.0],
        ...         [3, 4.0, 1.0, 2.0, -1.0],
        ...         [4, 5.0, 0.0, 2.0, -1.0]]
        >>> frame = tc.frame.create(data, schema)
        <progress>

        </hide>

    Consider the Frame *frame*, which contains the data:

        >>> frame.inspect()
         [#]  idnum  x1   x2   x3   x4
        ===============================
        [0]      0  1.0  4.0  0.0  -1.0
        [1]      1  2.0  3.0  0.0  -1.0
        [2]      2  3.0  2.0  1.0  -1.0
        [3]      3  4.0  1.0  2.0  -1.0
        [4]      4  5.0  0.0  2.0  -1.0

    Access covariance_matrix from the daal tk object (*d*).  covariance_matrix computes the covariance on each pair of
    columns in the user-provided list.

        >>> cov_matrix = tc.daaltk.operations.covariance_matrix(frame)
        <progress>

    The resulting table is:

        >>> cov_matrix.inspect()
        [#]  idnum  x1    x2    x3    x4
        =================================
        [0]    2.5   2.5  -2.5   1.5  0.0
        [1]    2.5   2.5  -2.5   1.5  0.0
        [2]   -2.5  -2.5   2.5  -1.5  0.0
        [3]    1.5   1.5  -1.5   1.0  0.0
        [4]    0.0   0.0   0.0   0.0  0.0

    """
    if not isinstance(frame, Frame):
        raise TypeError("frame must be a sparktk Frame, but received: %s" % type(frame))
    if data_column_names is not None:
        if not isinstance(data_column_names, list) or not all(isinstance(column, basestring) for column in data_column_names):
            raise TypeError("data_column_names must be a list of strings.")

    tc = frame._tc
    _scala_obj = _get_scala_obj(tc)
    result_frame = _scala_obj.covarianceMatrix(frame._scala,
                                               tc.jutils.convert.to_scala_option_list_string(data_column_names))
    return Frame(tc, result_frame)

def _get_scala_obj(tc):
    """Gets reference to the scala object"""
    return tc.sc._jvm.org.trustedanalytics.daaltk.operations.CovarianceMatrix