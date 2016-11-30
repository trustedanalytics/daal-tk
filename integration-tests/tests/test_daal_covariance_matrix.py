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

from setup import tc, rm
import logging
logger = logging.getLogger(__name__)

# Test frame schema and data
schema = [("idnum", int), ("x1", float), ("x2", float), ("x3", float), ("x4", float)]
data = [[0, 1.0, 4.0, 0.0, -1.0],[1, 2.0, 3.0, 0.0, -1.0],[2, 3.0, 2.0, 1.0, -1.0],[3, 4.0, 1.0, 2.0, -1.0],[4, 5.0, 0.0, 2.0, -1.0]]

def test_daal_covariance_matrix_invalid_params(tc):
    frame = tc.frame.create(data, schema)

    try:
        tc.daaltk.operations.covariance_matrix(None, frame.column_names)
    except Exception as e:
        assert("frame must be a sparktk Frame" in e.message)

    try:
        tc.daaltk.operations.covariance_matrix(frame, [1,2,3])
    except Exception as e:
        assert("data_column_names must be a list of strings" in e.message)

    try:
        tc.daaltk.operations.covariance_matrix(frame, "column")
    except Exception as e:
        assert("data_column_names must be a list of strings" in e.message)

def test_daal_covariance_matrix_columns(tc):
    frame = tc.frame.create(data, schema)

    # No column names should default to all columms
    all_cols = tc.daaltk.operations.covariance_matrix(frame)
    assert(all_cols.column_names == frame.column_names)

    # Limit columns
    cols = ["x1","x2","x3","x4"]
    four_cols = tc.daaltk.operations.covariance_matrix(frame, cols)
    assert(four_cols.column_names == cols)
