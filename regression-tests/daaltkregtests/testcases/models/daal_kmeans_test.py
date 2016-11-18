""" test cases for the Daal kmeans clustering algorithm """
import unittest
from daaltkregtests.lib import daaltk_test


class KMeansClustering(daaltk_test.DaalTKTestCase):

    def setUp(self):
        """Import the files to test against."""
        super(KMeansClustering, self).setUp()
        schema = [("Vec1", float),
                  ("Vec2", float),
                  ("Vec3", float),
                  ("Vec4", float),
                  ("Vec5", float),
                  ("term", str)]
        self.train_dataset = self.get_file("kmeans_train.csv")
        self.test_dataset = self.get_file("kmeans_test.csv")
        self.frame_train = self.context.frame.import_csv(
            self.train_dataset, schema=schema)
        self.frame_test = self.context.frame.import_csv(
            self.test_dataset, schema=schema)

    def test_kmeans_standard(self):
        """Tests standard usage of the kmeans cluster algorithm."""
        model = self.context.daaltk.models.clustering.kmeans.train(
            self.frame_train, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"], k=5)
    
        names = [u'Cluster:1', u'Cluster:0', u'Cluster:3',
                 u'Cluster:2', u'Cluster:4']
    
        centroids = sorted([[10.0, 10.0, 10.0, 10.0, 10.0],
                            [60.0, -70.0, -40.0, 30.0, 600.0],
                            [-10.0, -10.0, -10.0, -20.0, 10.0],
                            [-50.0, 70.0, -30.0, 90.0, 20.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])
    
        for i in range(5):
            self.assertEqual(model.cluster_sizes['Cluster:'+str(i)], 10000)
    
        for i1, i2 in zip(sorted(model.centroids.values()), centroids):
            for j1, j2 in zip(i1, i2):
                self.assertAlmostEqual(j1, j2, delta=.2)
    
        self.assertItemsEqual(model.centroids.keys(), names)
    
        test_frame = model.predict(self.frame_test)
        test_take = test_frame.to_pandas(test_frame.count())
        grouped = test_take.groupby(['predicted_cluster', 'term'])
        for i in grouped.size():
            self.assertEqual(10000, i)

    def test_column_weights(self):
        """Tests kmeans cluster algorithm with weighted values."""
        model = self.context.daaltk.models.clustering.kmeans.train(
            self.frame_train, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
            [0.01, 0.01, 0.01, 0.01, 0.01], 5)
    
        names = ['Cluster:1', u'Cluster:0', u'Cluster:3',
                 u'Cluster:2', u'Cluster:4']
    
        centroids = sorted([[.10, .10, .10, .10, .10],
                            [.60, -.70, -.40, .30, 6.0],
                            [-.10, -.10, -.10, -.20, .1],
                            [-.50, .70, -.30, .90, .20],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])
    
        for i in range(5):
            self.assertEqual(model.cluster_sizes['Cluster:'+str(i)], 10000)
    
        for i1, i2 in zip(sorted(model.centroids.values()), centroids):
            for j1, j2 in zip(i1, i2):
                self.assertAlmostEqual(j1, j2, delta=.2)
    
        self.assertItemsEqual(model.centroids.keys(), names)
    
        test_frame = model.predict(self.frame_test)
        test_take = test_frame.to_pandas(test_frame.count())
        grouped = test_take.groupby(['predicted_cluster', 'term'])
        for i in grouped.size():
            self.assertEqual(10000, i)

    def test_max_iterations(self):
        """Tests kmeans cluster algorithm with more iterations."""
        model = self.context.daaltk.models.clustering.kmeans.train(
            self.frame_train, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
            k=5, max_iterations=300)

        names = ['Cluster:1', u'Cluster:0', u'Cluster:3',
                  u'Cluster:2', u'Cluster:4']

        centroids = sorted([[10.0, 10.0, 10.0, 10.0, 10.0],
                            [60.0, -70.0, -40.0, 30.0, 600.0],
                            [-10.0, -10.0, -10.0, -20.0, 10.0],
                            [-50.0, 70.0, -30.0, 90.0, 20.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

        for i in range(5):
            self.assertEqual(model.cluster_sizes['Cluster:'+str(i)], 10000)

        for i1, i2 in zip(sorted(model.centroids.values()), centroids):
            for j1, j2 in zip(i1, i2):
                self.assertAlmostEqual(j1, j2, delta=.2)

        self.assertItemsEqual(model.centroids.keys(), names)

        test_frame = model.predict(self.frame_test)
        test_take = test_frame.to_pandas(test_frame.count())
        grouped = test_take.groupby(['predicted_cluster', 'term'])
        for i in grouped.size():
            self.assertEqual(10000, i)
 
    def test_different_columns(self):
        """Tests kmeans cluster algorithm with more iterations."""
        model = self.context.daaltk.models.clustering.kmeans.train(
            self.frame_train, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
            k=5, max_iterations=300)

        names = ['Cluster:1', u'Cluster:0', u'Cluster:3',
                 u'Cluster:2', u'Cluster:4']

        centroids = sorted([[10.0, 10.0, 10.0, 10.0, 10.0],
                            [60.0, -70.0, -40.0, 30.0, 600.0],
                            [-10.0, -10.0, -10.0, -20.0, 10.0],
                            [-50.0, 70.0, -30.0, 90.0, 20.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

        for i in range(5):
            self.assertEqual(model.cluster_sizes['Cluster:'+str(i)], 10000)

        for i1, i2 in zip(sorted(model.centroids.values()), centroids):
            for j1, j2 in zip(i1, i2):
                self.assertAlmostEqual(j1, j2, delta=.2)

        self.assertItemsEqual(model.centroids.keys(), names)

        self.frame_test.rename_columns(
            {"Vec1": 'Dim1', "Vec2": 'Dim2', "Vec3": "Dim3",
             "Vec4": "Dim4", "Vec5": 'Dim5'})
        test_frame = model.predict(
            self.frame_test, ['Dim1', 'Dim2', 'Dim3', 'Dim4', 'Dim5'])
        test_take = test_frame.to_pandas(test_frame.count())
        grouped = test_take.groupby(['predicted_cluster', 'term'])
        for i in grouped.size():
            self.assertEqual(10000, i)

    def test_predict_column(self):
        """Tests kmeans cluster algorithm with more iterations."""
        model = self.context.daaltk.models.clustering.kmeans.train(
            self.frame_train, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
            k=5, max_iterations=300)

        names = ['Cluster:1', u'Cluster:0', u'Cluster:3',
                 u'Cluster:2', u'Cluster:4']

        centroids = sorted([[10.0, 10.0, 10.0, 10.0, 10.0],
                            [60.0, -70.0, -40.0, 30.0, 600.0],
                            [-10.0, -10.0, -10.0, -20.0, 10.0],
                            [-50.0, 70.0, -30.0, 90.0, 20.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0]])

        for i in range(5):
            self.assertEqual(model.cluster_sizes['Cluster:'+str(i)], 10000)

        for i1, i2 in zip(sorted(model.centroids.values()), centroids):
            for j1, j2 in zip(i1, i2):
                self.assertAlmostEqual(j1, j2, delta=.2)

        self.assertItemsEqual(model.centroids.keys(), names)

        test_frame = model.predict(
            self.frame_test, label_column="predict_cluster")
        test_take = test_frame.to_pandas(test_frame.count())
        grouped = test_take.groupby(['predict_cluster', 'term'])
        for i in grouped.size():
            self.assertEqual(10000, i)

    def test_publish(self):
        """Tests kmeans cluster algorithm with random seeds."""
        model = self.context.daaltk.models.clustering.kmeans.train(self.frame_train,
                     ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
                     k=5)
        path = self.get_export_file(self.get_name("daal_kmeans"))
        model.export_to_mar(path)

        self.assertIn("hdfs", path)

    def test_max_iterations_negative(self):
        """Check error on negative number of iterations."""
        with(self.assertRaisesRegexp(Exception, "max iterations must be a positive value")):
            model = self.context.daaltk.models.clustering.kmeans.train(self.frame_train,
                         ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
                         k=5, max_iterations=-3)

    def test_max_iterations_bad_type(self):
        """Check error on a floating point number of iterations."""
        with(self.assertRaisesRegexp(Exception, "max_iterations must be an int")):
            model = self.context.daaltk.models.clustering.kmeans.train(self.frame_train,
                         ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"], max_iterations=[])

    def test_invalid_columns_predict(self):
        """Check error on a floating point number of iterations."""
        with(self.assertRaisesRegexp(Exception, "Invalid column name Vec1")):
            model = self.context.daaltk.models.clustering.kmeans.train(self.frame_train,
                         ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"])
            self.frame_test.rename_columns(
                {"Vec1": 'Dim1', "Vec2": 'Dim2',
                 "Vec3": "Dim3", "Vec4": "Dim4", "Vec5": 'Dim5'})
            model.predict(self.frame_test)

    def test_too_few_columns(self):
        """Check error on a floating point number of iterations."""
        with(self.assertRaisesRegexp(Exception, "Number of columns for train and predict should be same")):
            model = self.context.daaltk.models.clustering.kmeans.train(self.frame_train,
                         ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"])
            model.predict(self.frame_test, ["Vec1", "Vec2"])

    def test_k_negative(self):
        """Check error on negative number of clusters."""
        with(self.assertRaisesRegexp(Exception, "k must be at least 1")):
            model = self.context.daaltk.models.clustering.kmeans.train(self.frame_train,
                         ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"], k=-5)

    def test_k_bad_type(self):
        """Check error on float number of clusters."""
        with(self.assertRaisesRegexp(Exception, "k must be an int")):
            model = self.context.daaltk.models.clustering.kmeans.train(self.frame_train,
                         ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"], k=[])

    def test_null_frame(self):
        """Check error on null frame."""
        with(self.assertRaisesRegexp(Exception, "received: <type 'NoneType'>")):
            model = self.context.daaltk.models.clustering.kmeans.train(None, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"], k=5)


if __name__ == '__main__':
    unittest.main()
