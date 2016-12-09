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

"""tests daal covariance code"""
import unittest
from daaltkregtests.lib import daaltk_test


class DaalCovarianceTest(daaltk_test.DaalTKTestCase):

    def test_covar_matrix(self):
        """test daal covariance matrix"""
        data_in = self.get_file("covariance_correlation.csv")
        base_frame = self.context.frame.import_csv(data_in)

        # use daaltk to calc expected result
        # we will use sparktk result for reference
        # since the algorithm should essentially be the same
        # we expect the daaltk/sparktk results to match
        covar_matrix = self.context.daaltk.operations.covariance_matrix(base_frame)
        sparktk_result = base_frame.covariance_matrix(base_frame.column_names)
       
        # flatten the expected and actual result for ease of comparison
        covar_flat = [item for sublist in covar_matrix.take(covar_matrix.count()) for item in sublist]
        sparktk_res_flat = [item for sublist in sparktk_result.take(sparktk_result.count()) for item in sublist]
        
        # compare expected and actual result
        for daal_value, spark_value in zip(covar_flat, sparktk_res_flat):
            self.assertAlmostEqual(daal_value, spark_value, 7)
   

if __name__ == "__main__":
    unittest.main()
