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

"""Tests Naive Bayes Model against known values"""
import unittest
import math
from daaltkregtests.lib import daaltk_test


class DaalNaiveBayes(daaltk_test.DaalTKTestCase):

    def setUp(self):
        """Build the frames needed for the tests."""
        super(DaalNaiveBayes, self).setUp()

        dataset = self.get_file("naive_bayes_5_col.csv")
        schema = [("label", int),
                  ("count", int),
                  ("f1", int),
                  ("f2", int),
                  ("f3", int)]
        self.frame = self.context.frame.import_csv(dataset, schema=schema)

    def test_model_train_empty_feature(self):
        """Test empty string for training features throws errors."""
        with(self.assertRaisesRegexp(Exception, "observation_columns must be a list of strings")):
            self.context.daaltk.models.classification.naive_bayes.train(self.frame, "label", "")

    def test_model_train_empty_label_coloum(self):
         """Test empty string for label coloum throws error."""
         with(self.assertRaisesRegexp(Exception, "label column must not be null nor empty")):
             self.context.daaltk.models.classification.naive_bayes.train(self.frame, "", ['f1', 'f2', 'f3'])
 
    def test_model_test(self):
         """Test training intializes theta, pi and labels"""
         model = self.context.daaltk.models.classification.naive_bayes.train(self.frame, "label", ['f1', 'f2', 'f3'], 3)
         values = sorted(map(
             lambda x: map(math.exp, x), model.feature_log_prob))
         baseline = sorted([[0.3, 0.2, 0.5], [0.7, 0.0, 0.3], [0.2, 0.6, 0.2]])
         for i, j in zip(values, baseline):
             for k, l in zip(i, j):
                 self.assertAlmostEqual(k, l, delta=.05)
 
         # This is hacky, should really train on another
         # dataset
         res = model.test(self.frame, "label", ['f1', 'f2', 'f3'])
         self.assertGreater(res.precision, 0.9)
         self.assertGreater(res.recall, 0.9)
         self.assertGreater(res.accuracy, 0.9)
         self.assertGreater(res.f_measure, 0.9)
 
    def test_model_publish_bayes(self):
        """Test training intializes theta, pi and labels"""
        model = self.context.daaltk.models.classification.naive_bayes.train(self.frame, "label", ['f1', 'f2', 'f3'])
        path = self.get_export_file(self.get_name("daal_naive_bayes"))
        model.export_to_mar(path)
        self.assertIn("hdfs", path)
 
    def test_model_test_paramater_initiation(self):
        """Test training intializes theta, pi and labels"""
        model = self.context.daaltk.models.classification.naive_bayes.train(self.frame, "label", ['f1', 'f2', 'f3'], 3)

        # This is hacky, should really train on another
        # dataset
        res = model.predict(self.frame, ['f1', 'f2', 'f3'])
        res.add_columns(lambda x: [int(x['predicted_class'])], [("pc", int)])
        res_2 = res.multiclass_classification_metrics('label', 'pc')
        self.assertGreater(res_2.precision, 0.9)
        self.assertGreater(res_2.recall, 0.9)
        self.assertGreater(res_2.accuracy, 0.9)
        self.assertGreater(res_2.f_measure, 0.9)


if __name__ == '__main__':
    unittest.main()
