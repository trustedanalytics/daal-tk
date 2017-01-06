""" Tests Linear Regression scoring engine """
import unittest
import os
from daaltkregtests.lib import daaltk_test
from daaltkregtests.lib import score_utils


class LinearRegression(daaltk_test.DaalTKTestCase):

    def setUp(self):
        """Build test frame"""
        super(LinearRegression, self).setUp()
        dataset = self.get_file("linear_regression_gen.csv")
        schema = [("c1", float),
                  ("c2", float),
                  ("c3", float),
                  ("c4", float),
                  ("label", float)]

        self.frame = self.context.frame.import_csv(
            dataset, schema=schema)

    def test_model_scoring(self):
        """Test publishing a linear regression model"""
        model = self.context.daaltk.models.regression.linear_regression.train(self.frame, "label", ['c1', 'c2', 'c3', 'c4'])

        predict = model.predict(self.frame, ['c1', 'c2', 'c3', 'c4'])
        test_rows = predict.to_pandas(predict.count())

        file_name = self.get_name("linear_regression")
        model_path = model.export_to_mar(self.get_export_file(file_name))
        with score_utils.scorer(
                model_path, self.id()) as scorer:
            for index, row in test_rows.iterrows():
                res = scorer.score(
                    [dict(zip(["c1", "c2", "c3", "c4"], list(row[0:4])))])
                self.assertAlmostEqual(
                    row["predict_label"], res.json()["data"][0]['score'])

            


if __name__ == '__main__':
    unittest.main()
