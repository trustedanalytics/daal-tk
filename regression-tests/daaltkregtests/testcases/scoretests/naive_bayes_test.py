""" Tests Naive Bayes Model against known values.  """
import unittest
import os
from daaltkregtests.lib import daaltk_test
from daaltkregtests.lib import score_utils


class NaiveBayes(daaltk_test.DaalTKTestCase):

    def setUp(self):
        """Build the frames needed for the tests."""
        super(NaiveBayes, self).setUp()

        dataset = self.get_file("naive_bayes.csv")
        schema = [("label", int),
                  ("f1", int),
                  ("f2", int),
                  ("f3", int)]
        self.frame = self.context.frame.import_csv(dataset, schema=schema)

    def test_model_scoring(self):
        """Test training intializes theta, pi and labels"""
        model = self.context.daaltk.models.classification.naive_bayes.train(self.frame, "label", ['f1', 'f2', 'f3'])

        res = model.predict(self.frame, ['f1', 'f2', 'f3'])

        analysis = res.to_pandas()
        file_name = self.get_name("daal_naive_bayes")
        model_path = model.export_to_mar(self.get_export_file(file_name))
        with score_utils.scorer(
                model_path, self.id()) as scorer:
            for index, row in analysis.iterrows():
                r = scorer.score(
                    [dict(zip(['f1', 'f2', 'f3'],
                    map(lambda x: int(x), (row[1:4]))))])
                self.assertEqual(
                    r.json()["data"][0]['score'], row.predicted_class)


if __name__ == '__main__':
    unittest.main()
