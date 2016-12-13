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

"""Test DAAL PCA implementation. It is unknown what this functionality does"""
import unittest
from daaltkregtests.lib import daaltk_test


class DaalPrincipalComponent(daaltk_test.DaalTKTestCase):
    # expected singular values
    expected_singular_val = [3373.70412657, 594.11385671,
                             588.713470217, 584.157023124,
                             579.433395835, 576.659495077,
                             572.267630461, 568.224352464,
                             567.328732759, 560.882281619]
    # expected right-singular vectors V
    expected_R_singular_vec = \
        [[0.315533916, -0.3942771, 0.258362247, -0.0738539198,
          -0.460673735, 0.0643077298, -0.0837131184, 0.0257963888,
          0.00376728499, 0.669876972],
         [0.316500921, -0.165508013, -0.131017612, 0.581988787,
          -0.0863507191, 0.160473134, 0.53134635, 0.41199152,
          0.0823770991, -0.156517367],
         [0.316777341, 0.244415549, 0.332413311, -0.377379981,
          0.149653873, 0.0606339992, -0.163748261, 0.699502817,
          -0.171189721, -0.124509149],
         [0.318988109, -0.171520719, -0.250278714, 0.335635209,
          0.580901954, 0.160427725, -0.531610364, -0.0304943121,
          -0.0785743304, 0.201591811],
         [0.3160833, 0.000386702461, -0.108022985, 0.167086405,
          -0.470855879, -0.256296677, -0.318727111, -0.155621638,
          -0.521547782, -0.418681224],
         [0.316721742, 0.288319245, 0.499514144, 0.267566455,
          -0.0338341451, -0.134086469, -0.184724393, -0.246523528,
          0.593753078, -0.169969303],
         [0.315335647, -0.258529064, 0.374780341, -0.169762381,
          0.416093803, -0.118232778, 0.445019707, -0.395962728,
          -0.337229123, -0.0937071881],
         [0.314899154, -0.0294147958, -0.447870311, -0.258339192,
          0.0794841625, -0.71141762, 0.110951688, 0.102784186,
          0.292018251, 0.109836478],
         [0.315542865, -0.236497774, -0.289051199, -0.452795684,
          -0.12175352, 0.5265342, -0.0312645934, -0.180142504,
          0.318334436, -0.359303747],
         [0.315875856, 0.72196434, -0.239088332, -0.0259999274,
          -0.0579153559, 0.244335633, 0.232808362, -0.233600306,
          -0.181191102, 0.3413174]]

    def setUp(self):
        super(DaalPrincipalComponent, self).setUp()
        schema = [("X1", int),
                   ("X2", int),
                   ("X3", int),
                   ("X4", int),
                   ("X5", int),
                   ("X6", int),
                   ("X7", int),
                   ("X8", int),
                   ("X9", int),
                   ("X10", int)]
        training_data = self.get_file("pcadata.csv")
        self.frame = self.context.frame.import_csv(training_data, schema=schema)

    @unittest.skip("daaltk: pca_model has no singular vectors")
    def test_daal_principal_components_train_mean(self):
        """Test the train functionality with mean centering"""
        pca_train_out = self.context.daaltk.models.dimensionality_reduction.principal_components.train(self.frame,
                                       ["X1", "X2", "X3", "X4", "X5",
                                        "X6", "X7", "X8", "X9", "X10"],
                                       True, 10)

        # actual right-singular vectors
        actual_R_singular_vec = pca_train_out.right_singular_vectors

        # actual singular values
        actual_singular_val = pca_train_out.singular_values
        for c in self.frame.column_names:
            mean = self.frame.column_summary_statistics(c)["mean"]
            self.frame.add_columns(
                lambda x: x[c] - mean, (c+"_n", float))

        pcamodelmean = self.context.daaltk.models.dimensionality_reduction.principal_components.train(
            self.frame,
            ["X1_n", "X2_n", "X3_n", "X4_n", "X5_n",
             "X6_n", "X7_n", "X8_n", "X9_n", "X10_n"],
            False, 10)

        # actual right-singular vectors
        actual_R_singular_vec_mean = pca_train_out.right_singular_vectors
        # actual singular values
        actual_singular_val_mean = pca_train_out.singular_values

        expected_actual = zip(actual_singular_val, actual_singular_val_mean)
        for expected, actual in expected_actual:
            self.assertAlmostEqual(expected, actual, 8)

        expected_actual = zip(
            actual_R_singular_vec, actual_R_singular_vec_mean)
        for expected, actual in expected_actual:
            for f1, f2 in zip(expected, actual):
                self.assertAlmostEqual(f1, f2, 4)

    @unittest.skip("daaltk: pca_model has no singular vectors")
    def test_daal_pca_predict(self):
        """Test the train functionality"""
        pca_train_out = self.context.daaltk.models.dimensionality_reduction.principal_components.train(self.frame,
                                       ["X1", "X2", "X3", "X4", "X5",
                                        "X6", "X7", "X8", "X9", "X10"],
                                       False, 10)

        pca_train_out.predict(self.frame, False)
        pd_frame = self.frame.to_pandas(self.frame.count())
        actual_R_singular_vec = map(
            list, zip(*pca_train_out.right_singular_vectors))
        for _, i in pd_frame.iterrows():
            vec1 = i[0:10]
            vec2 = i[10:]
            dot_product = [sum([(r1)*(r2) for r1, r2 in zip(vec1, k)])
                           for k in actual_R_singular_vec]
            for i, j in zip(vec2, dot_product):
                self.assertAlmostEqual(i, j)

    @unittest.skip("daaltk: pca_model has no singular vectors")
    def test_daal_pca_train(self):
        """Test the train functionality"""
        pca_train_out = self.context.daaltk.models.dimensionality_reduction.principal_components.train(self.frame,
                                       ["X1", "X2", "X3", "X4", "X5",
                                        "X6", "X7", "X8", "X9", "X10"],
                                       False, 10)

        # actual right-singular vectors
        actual_R_singular_vec = pca_train_out.right_singular_vectors

        # actual singular values
        actual_singular_val = pca_train_out.singular_values

        expected_actual = zip(self.expected_singular_val, actual_singular_val)
        for expected, actual in expected_actual:
            self.assertAlmostEqual(expected, actual, 8)

        expected_actual = zip(actual_R_singular_vec,
                              self.expected_R_singular_vec)
        for expected, actual in expected_actual:
            for f1, f2 in zip(expected, actual):
                self.assertAlmostEqual(abs(f1), abs(f2), 4)

    def test_daal_pca_publish(self):
        """Test the publish functionality"""
        pcamodel = self.context.daaltk.models.dimensionality_reduction.principal_components.train(
            self.frame,
            ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"],
            False, 10)
        path = self.get_export_file(self.get_name("daaltk_pca"))
        pcamodel.export_to_mar(path)
        self.assertIn("hdfs", path)

    @unittest.skip("daaltk: pca_model has no singular vectors")
    def test_daal_pca_default(self):
        """Test default no. of k"""
        pca_train_out = self.context.daaltk.models.dimensionality_reduction.principal_components.train(
            self.frame,
            ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"],
            False)
        # actual right-singular vectors
        actual_R_singular_vec = pca_train_out.right_singular_vectors

        # actual singular values
        actual_singular_val = pca_train_out.singular_values

        for ind in xrange(0, len(actual_singular_val)):
                self.assertAlmostEqual(round(actual_singular_val[ind], 8),
                                       self.expected_singular_val[ind])

        for ind in xrange(0, len(actual_R_singular_vec)):
            for ind2 in xrange(0, len(actual_R_singular_vec[ind])):
                self.assertEqual(
                    abs(round(actual_R_singular_vec[ind][ind2], 6)),
                    abs(round(self.expected_R_singular_vec[ind][ind2], 6)))

    def test_daal_pca_bad_no_of_k(self):
        """Test invalid k value in train"""
        with self.assertRaisesRegexp(Exception, "k must be less than or equal to number of observation columns"):
                self.context.daaltk.models.dimensionality_reduction.principal_components.train(self.frame,
                               ["X1", "X2", "X3", "X4", "X5",
                                "X6", "X7", "X8", "X9", "X10"],
                               k=11)

    def test_daal_pca_invalid_k(self):
        """Test k < 1 in train"""
        with self.assertRaisesRegexp(Exception, "k must be smaller than the number of observation columns"):
                self.context.daaltk.models.dimensionality_reduction.principal_components.train(self.frame,
                               ["X1", "X2", "X3", "X4", "X5",
                                "X6", "X7", "X8", "X9", "X10"],
                               k=0)

    def test_daal_pca_bad_column_name(self):
        """Test bad feature column name"""
        with self.assertRaisesRegexp(Exception, "column ERR was not found"):
                self.context.daaltk.models.dimensionality_reduction.principal_components.train(self.frame,
                               ["ERR", "X2", "X3", "X4", "X5",
                                "X6", "X7", "X8", "X9", "X10"],
                               k=10)

    def test_daal_pca_bad_column_type(self):
        """Test bad feature column name type"""
        with self.assertRaisesRegexp(Exception, "columns must be a list of strings"):
                self.context.daaltk.models.dimensionality_reduction.principal_components.train(self.frame, 10, k=10)


if __name__ == '__main__':
    unittest.main()
