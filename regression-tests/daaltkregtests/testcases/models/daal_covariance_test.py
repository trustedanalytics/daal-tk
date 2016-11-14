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
