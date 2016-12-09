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

""" Tests Linear Regression Model, with known coefficents """
import unittest
from daaltkregtests.lib import daaltk_test


class DaalLinearRegression(daaltk_test.DaalTKTestCase):

    def setUp(self):
        super(DaalLinearRegression, self).setUp()
        dataset = self.get_file("linear_regression_gen.csv")
        schema = [("c1", float),
                  ("c2", float),
                  ("c3", float),
                  ("c4", float),
                  ("label", float)]

        self.frame = self.context.frame.import_csv(dataset, schema=schema)

    def test_model_test(self):
        """Test daal linear regression publish"""
        model = self.context.daaltk.models.regression.linear_regression.train(self.frame, "label", ['c1', 'c2', 'c3', 'c4'])
        baseline = { u'mean_squared_error': 0,
                     u'r_2': 1.0,
                     u'mean_absolute_error': 0,
                     u'explained_variance': 3,
                     u'root_mean_squared_error': 0}

        res = model.test(self.frame)
        self.assertAlmostEqual(res.mean_squared_error, 0)
        self.assertAlmostEqual(res.r2, 1.0)
        self.assertAlmostEqual(res.mean_absolute_error, 0)
        self.assertAlmostEqual(res.explained_variance, 3.0, delta=0.1)
        self.assertAlmostEqual(res.root_mean_squared_error, 0)

    def test_model_publish(self):
        """Test daal linear regression publish"""
        model = self.context.daaltk.models.regression.linear_regression.train(self.frame, "label", ['c1', 'c2', 'c3', 'c4'])
        path = self.get_export_file(self.get_name("daaltk_linear_regression"))
        model.export_to_mar(path)
        self.assertIn("hdfs", path)

    def test_model_predict_output(self):
        """Test daal linear regression output and predict is correct"""
        model = self.context.daaltk.models.regression.linear_regression.train(
            self.frame, "label", ['c1', 'c2', 'c3', 'c4'])

        for i, j in zip([0.5, -0.7, -0.24, 0.4], model.weights):
            self.assertAlmostEqual(i, j, places=4)

        res = model.predict(self.frame, ['c1', 'c2', 'c3', 'c4'])
        pd_res = res.to_pandas(res.count())
        for _, i in pd_res.iterrows():
            self.assertAlmostEqual(i["label"], i["predict_label"])


if __name__ == '__main__':
    unittest.main()
