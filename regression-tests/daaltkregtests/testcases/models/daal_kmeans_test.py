""" test cases for the Daal kmeans clustering algorithm """
import unittest

from daaltkregtests.lib import sparktk_test


class KMeansClustering(sparktk_test.SparkTKTestCase):

    def setUp(self):
        """Import the files to test against."""
        super(KMeansClustering, self).setUp()
        schema = [("Vec1", float),
                  ("Vec2", float),
                  ("Vec3", float),
                  ("Vec4", float),
                  ("Vec5", float),
                  ("term", str)]

        self.frame_train = self.context.frame.import_csv(
            self.get_file("kmeans_train.csv"), schema=schema)
        self.frame_test = self.context.frame.import_csv(
            self.get_file("kmeans_test.csv"), schema=schema)

    def test_kmeans_standard(self):
        """Tests standard usage of the kmeans cluster algorithm."""
        model = self.context.daaltk.models.clustering.kmeans.train(
            self.frame_train, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"], k=5)
    #
    #    names = [u'Cluster:1', u'Cluster:0', u'Cluster:3',
    #             u'Cluster:2', u'Cluster:4']
    #
    #    centroids = sorted([[10.0, 10.0, 10.0, 10.0, 10.0],
    #                        [60.0, -70.0, -40.0, 30.0, 600.0],
    #                        [-10.0, -10.0, -10.0, -20.0, 10.0],
    #                        [-50.0, 70.0, -30.0, 90.0, 20.0],
    #                        [0.0, 0.0, 0.0, 0.0, 0.0]])
    #
    #    for i in range(5):
    #        self.assertEqual(result['cluster_size']['Cluster:'+str(i)], 10000)
    #
    #    for i1, i2 in zip(sorted(result['centroids'].values()), centroids):
    #        for j1, j2 in zip(i1, i2):
    #            self.assertAlmostEqual(j1, j2, delta=.2)
    #
    #    self.assertItemsEqual(result['centroids'].keys(), names)
    #
    #    test_frame = kmodel.predict(self.frame_test)
    #    test_take = test_frame.download(test_frame.row_count)
    #    grouped = test_take.groupby(['predicted_cluster', 'term'])
    #    for i in grouped.size():
    #        self.assertEqual(10000, i)

    #def test_column_weights(self):
    #    """Tests kmeans cluster algorithm with weighted values."""
    #    kmodel = ia.DaalKMeansModel()
    #
    #    result = kmodel.train(
    #        self.frame_train, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
    #        [0.01, 0.01, 0.01, 0.01, 0.01], 5)
    #
    #    names = ['Cluster:1', u'Cluster:0', u'Cluster:3',
    #             u'Cluster:2', u'Cluster:4']
    #
    #    centroids = sorted([[.10, .10, .10, .10, .10],
    #                        [.60, -.70, -.40, .30, 6.0],
    #                        [-.10, -.10, -.10, -.20, .1],
    #                        [-.50, .70, -.30, .90, .20],
    #                        [0.0, 0.0, 0.0, 0.0, 0.0]])
    #
    #    for i in range(5):
    #        self.assertEqual(result['cluster_size']['Cluster:'+str(i)], 10000)
    #
    #    for i1, i2 in zip(sorted(result['centroids'].values()), centroids):
    #        for j1, j2 in zip(i1, i2):
    #            self.assertAlmostEqual(j1, j2, delta=.2)
    #
    #    self.assertItemsEqual(result['centroids'].keys(), names)
    #
    #    test_frame = kmodel.predict(self.frame_test)
    #    test_take = test_frame.download(test_frame.row_count)
    #    grouped = test_take.groupby(['predicted_cluster', 'term'])
    #    for i in grouped.size():
    #        self.assertEqual(10000, i)

    #def test_max_iterations(self):
    #    """Tests kmeans cluster algorithm with more iterations."""
    #    kmodel = ia.DaalKMeansModel()
    #
    #    result = kmodel.train(
    #        self.frame_train, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
    #        k=5, max_iterations=300)
# 
#         names = ['Cluster:1', u'Cluster:0', u'Cluster:3',
#                  u'Cluster:2', u'Cluster:4']
# 
#         centroids = sorted([[10.0, 10.0, 10.0, 10.0, 10.0],
#                             [60.0, -70.0, -40.0, 30.0, 600.0],
#                             [-10.0, -10.0, -10.0, -20.0, 10.0],
#                             [-50.0, 70.0, -30.0, 90.0, 20.0],
#                             [0.0, 0.0, 0.0, 0.0, 0.0]])
# 
#         for i in range(5):
#             self.assertEqual(result['cluster_size']['Cluster:'+str(i)], 10000)
# 
#         for i1, i2 in zip(sorted(result['centroids'].values()), centroids):
#             for j1, j2 in zip(i1, i2):
#                 self.assertAlmostEqual(j1, j2, delta=.2)
# 
#         self.assertItemsEqual(result['centroids'].keys(), names)
# 
#         test_frame = kmodel.predict(self.frame_test)
#         test_take = test_frame.download(test_frame.row_count)
#         grouped = test_take.groupby(['predicted_cluster', 'term'])
#         for i in grouped.size():
#             self.assertEqual(10000, i)
# 
#     def test_different_columns(self):
#         """Tests kmeans cluster algorithm with more iterations."""
#         kmodel = ia.DaalKMeansModel()
# 
#         result = kmodel.train(
#             self.frame_train, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
#             k=5, max_iterations=300)
# 
#         names = ['Cluster:1', u'Cluster:0', u'Cluster:3',
#                  u'Cluster:2', u'Cluster:4']
# 
#         centroids = sorted([[10.0, 10.0, 10.0, 10.0, 10.0],
#                             [60.0, -70.0, -40.0, 30.0, 600.0],
#                             [-10.0, -10.0, -10.0, -20.0, 10.0],
#                             [-50.0, 70.0, -30.0, 90.0, 20.0],
#                             [0.0, 0.0, 0.0, 0.0, 0.0]])
# 
#         for i in range(5):
#             self.assertEqual(result['cluster_size']['Cluster:'+str(i)], 10000)
# 
#         for i1, i2 in zip(sorted(result['centroids'].values()), centroids):
#             for j1, j2 in zip(i1, i2):
#                 self.assertAlmostEqual(j1, j2, delta=.2)
# 
#         self.assertItemsEqual(result['centroids'].keys(), names)
# 
#         self.frame_test.rename_columns(
#             {"Vec1": 'Dim1', "Vec2": 'Dim2', "Vec3": "Dim3",
#              "Vec4": "Dim4", "Vec5": 'Dim5'})
#         test_frame = kmodel.predict(
#             self.frame_test, ['Dim1', 'Dim2', 'Dim3', 'Dim4', 'Dim5'])
#         test_take = test_frame.download(test_frame.row_count)
#         grouped = test_take.groupby(['predicted_cluster', 'term'])
#         for i in grouped.size():
#             self.assertEqual(10000, i)
# 
#     @unittest.skip("this doesn't work")
#     def test_predict_column(self):
#         """Tests kmeans cluster algorithm with more iterations."""
#         kmodel = ia.DaalKMeansModel()
# 
#         result = kmodel.train(
#             self.frame_train, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
#             5, max_iterations=300)
# 
#         names = ['Cluster:1', u'Cluster:0', u'Cluster:3',
#                  u'Cluster:2', u'Cluster:4']
# 
#         centroids = sorted([[10.0, 10.0, 10.0, 10.0, 10.0],
#                             [60.0, -70.0, -40.0, 30.0, 600.0],
#                             [-10.0, -10.0, -10.0, -20.0, 10.0],
#                             [-50.0, 70.0, -30.0, 90.0, 20.0],
#                             [0.0, 0.0, 0.0, 0.0, 0.0]])
# 
#         for i in range(5):
#             self.assertEqual(result['cluster_sizes']['Cluster:'+str(i)], 10000)
# 
#         for i1, i2 in zip(sorted(result['centroids'].values()), centroids):
#             for j1, j2 in zip(i1, i2):
#                 self.assertAlmostEqual(j1, j2, delta=.2)
# 
#         self.assertItemsEqual(result['centroids'].keys(), names)
# 
#         test_frame = kmodel.predict(
#             self.frame_test, label_column="predict_cluster")
#         test_take = test_frame.download(test_frame.row_count)
#         grouped = test_take.groupby(['predict_cluster', 'term'])
#         for i in grouped.size():
#             self.assertEqual(10000, i)
# 
#     def test_publish(self):
#         """Tests kmeans cluster algorithm with random seeds."""
#         kmodel = ia.DaalKMeansModel()
# 
#         kmodel.train(self.frame_train,
#                      ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
#                      k=5)
#         path = kmodel.publish()
# 
#         self.assertIn("hdfs", path['sourceUri'])
#         self.assertIn("tar", path['sourceUri'])
#         self.assertIn("hdfs", path['targetUri'])
#         self.assertIn("tar", path['targetUri'])
#         self.assertIn('size', path.keys())
#         self.assertEqual(path['dataSample'], '')
#         self.assertEqual(path['format'], 'tar')
#         self.assertFalse(path['isPublic'])
#         self.assertEqual(path['recordCount'], 0)
#         self.assertEqual(path['category'], 'model')
# 
#     def test_max_iterations_negative(self):
#         """Check error on negative number of iterations."""
#         with(self.assertRaises(ia.rest.command.CommandServerError)):
#             kmodel = ia.DaalKMeansModel()
# 
#             kmodel.train(self.frame_train,
#                          ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
#                          5, max_iterations=-3)
# 
#     def test_max_iterations_bad_type(self):
#         """Check error on a floating point number of iterations."""
#         with(self.assertRaises(ia.rest.command.CommandServerError)):
#             kmodel = ia.DaalKMeansModel()
# 
#             kmodel.train(self.frame_train,
#                          ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
#                          5, max_iterations=[])
# 
#     def test_invalid_columns_predict(self):
#         """Check error on a floating point number of iterations."""
#         with(self.assertRaises(ia.rest.command.CommandServerError)):
#             kmodel = ia.DaalKMeansModel()
# 
#             kmodel.train(self.frame_train,
#                          ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
#                          5, max_iterations=[])
#             self.frame_test.rename_columns(
#                 {"Vec1": 'Dim1', "Vec2": 'Dim2',
#                  "Vec3": "Dim3", "Vec4": "Dim4", "Vec5": 'Dim5'})
#             kmodel.predict(self.frame_test)
# 
#     def test_too_few_columns(self):
#         """Check error on a floating point number of iterations."""
#         with(self.assertRaises(ia.rest.command.CommandServerError)):
#             kmodel = ia.DaalKMeansModel()
# 
#             kmodel.train(self.frame_train,
#                          ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"],
#                          5, max_iterations=[])
#             kmodel.predict(self.frame_test, ["Vec1", "Vec2"])
# 
#     def test_k_negative(self):
#         """Check error on negative number of clusters."""
#         with(self.assertRaises(ia.rest.command.CommandServerError)):
#             kmodel = ia.DaalKMeansModel()
# 
#             kmodel.train(self.frame_train,
#                          ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"], -5)
# 
#     def test_k_bad_type(self):
#         """Check error on float number of clusters."""
#         with(self.assertRaises(ia.rest.command.CommandServerError)):
#             kmodel = ia.DaalKMeansModel()
# 
#             kmodel.train(self.frame_train,
#                          ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"], [])
# 
#     def test_null_frame(self):
#         """Check error on null frame."""
#         with(self.assertRaises(ia.rest.command.CommandServerError)):
#             kmodel = ia.DaalKMeansModel()
# 
#             kmodel.train(None, ["Vec1", "Vec2", "Vec3", "Vec4", "Vec5"], 5)


if __name__ == '__main__':
    unittest.main()
