/**
 *  Copyright (c) 2016 Intel Corporation 
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package org.trustedanalytics.daaltk.models.clustering.kmeans

import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.Row
import org.scalatest.Matchers
import org.trustedanalytics.daaltk.testutils.TestingSparkContextWordSpec
import org.trustedanalytics.sparktk.frame.{ FrameSchema, Frame, DataTypes, Column }

class KMeansModelTest extends TestingSparkContextWordSpec with Matchers {

  val frameData: Array[Row] = Array(
    new GenericRow(Array[Any](2.0, "ab")),
    new GenericRow(Array[Any](1.0, "cd")),
    new GenericRow(Array[Any](7.0, "ef")),
    new GenericRow(Array[Any](1.0, "gh")),
    new GenericRow(Array[Any](9.0, "ij")),
    new GenericRow(Array[Any](2.0, "kl")),
    new GenericRow(Array[Any](0.0, "mn")),
    new GenericRow(Array[Any](6.0, "op")),
    new GenericRow(Array[Any](5.0, "qr")),
    new GenericRow(Array[Any](120.0, "outlier")))
  val schema = new FrameSchema(List(Column("data", DataTypes.float32), Column("name", DataTypes.string)))

  "KMeansModel" should {

    "create a DAAL KMeansModel, train, and predict" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // model train
      val model = KMeansModel.train(frame, List("data"), None, k = 2, maxIterations = 20)
      model shouldBe a[KMeansModel]

      // check trained model properties
      assert(model.clusters.length == 2)
      assert(model.centroids.length == 2)

      // we should have one cluster of 1 and one cluster of 9
      assert(model.clusterSize.sameElements(Array(1, 9)))
      val outlierClusterIndex = model.clusterSize.indexOf(1) // get index of the one outlier cluster
      val groupClusterIndex = model.clusterSize.indexOf(9) // get index of the group cluster

      // check centroid locations
      assert(model.centroids(outlierClusterIndex)(0) == 120.0)
      assertAlmostEqual(model.centroids(groupClusterIndex)(0), 3.6666666666666665)

      // model predict
      val predictFrame = model.predict(frame, Some(List("data")))
      assert(predictFrame.schema.hasColumn("distance_from_cluster_0"))
      assert(predictFrame.schema.hasColumn("distance_from_cluster_1"))
      assert(predictFrame.schema.hasColumn("predicted_cluster"))
      assert(predictFrame.rowCount() == 10)
      val predictData = predictFrame.take(10)
      val clusterIndex = predictFrame.schema.columnIndex("predicted_cluster")
      val predictedClusters = predictData.map(row => row.getInt(clusterIndex))
      assert(predictedClusters.sameElements(Array(groupClusterIndex, groupClusterIndex, groupClusterIndex,
        groupClusterIndex, groupClusterIndex, groupClusterIndex, groupClusterIndex, groupClusterIndex,
        groupClusterIndex, outlierClusterIndex)))
    }

    "throw an exception for invalid train arguments" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // null frame
      var thrown = the[Exception] thrownBy KMeansModel.train(null, List("data"), None, k = 2, maxIterations = 20)
      assert(thrown.getMessage.contains("frame is required"))

      // invalid observation column
      thrown = the[Exception] thrownBy KMeansModel.train(frame, List("bogus"), None, k = 2, maxIterations = 20)
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))

      // invalid number of column scalings
      thrown = the[Exception] thrownBy KMeansModel.train(frame, List("data"), Some(List[Double](1.0, 2.5, 7.0, 14.3)))
      assert(thrown.getMessage.contains("column scalings must be empty or the same size as observation columns"))

      // invalid k
      thrown = the[Exception] thrownBy KMeansModel.train(frame, List("data"), None, k = 0, maxIterations = 20)
      assert(thrown.getMessage.contains("k must be at least 1"))

      // invalid max iterations
      thrown = the[Exception] thrownBy KMeansModel.train(frame, List("data"), None, k = 2, maxIterations = 0)
      assert(thrown.getMessage.contains("max iterations must be a positive value"))
    }

    "throw an exception for invalid predict arguments" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)
      val model = KMeansModel.train(frame, List("data"), None, k = 2, maxIterations = 20)

      // null frame
      var thrown = the[Exception] thrownBy model.predict(null, Some(List("data")))
      assert(thrown.getMessage.contains("frame is required"))

      // invalid observation column
      thrown = the[Exception] thrownBy model.predict(frame, Some(List("bogus")))
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))
    }

    "return a prediction when calling score on a trained model" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // model train
      val model = KMeansModel.train(frame, List("data"), None, k = 2, maxIterations = 20)

      // Find out which cluster has the majority (we will score with a value in that group)
      val clusterIndex = model.clusterSize.indexOf(9)
      val expectedCluster = model.clusters(clusterIndex).split(":")(1).toInt

      // Score
      val inputData = 2.0
      val inputArray = Array[Any](inputData)
      assert(model.input().length == inputArray.length)
      val scoreResult = model.score(inputArray)
      assert(scoreResult.length == model.output().length)
      assert(scoreResult(0) == inputData)
      scoreResult(1) match {
        case prediction: Integer => assert(prediction == expectedCluster)
        case _ => throw new RuntimeException(s"Expected prediction to be an Integer but is ${scoreResult(1).getClass.getSimpleName}")
      }
    }
  }

}
