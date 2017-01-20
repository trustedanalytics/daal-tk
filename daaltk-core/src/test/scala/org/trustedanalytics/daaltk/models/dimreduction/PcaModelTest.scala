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
package org.trustedanalytics.daaltk.models.dimreduction

import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.Row
import org.scalatest.Matchers
import org.trustedanalytics.daaltk.models.dimreduction.pca.PcaModel
import org.trustedanalytics.daaltk.testutils.TestingSparkContextWordSpec
import org.trustedanalytics.sparktk.frame.{ Column, DataTypes, Frame, FrameSchema }

class PcaModelTest extends TestingSparkContextWordSpec with Matchers {

  val frameData: Array[Row] = Array(
    new GenericRow(Array[Any](2.6, 1.7, 0.3, 1.5, 0.8, 0.7)),
    new GenericRow(Array[Any](3.3, 1.8, 0.4, 0.7, 0.9, 0.8)),
    new GenericRow(Array[Any](3.5, 1.7, 0.3, 1.7, 0.6, 0.4)),
    new GenericRow(Array[Any](3.7, 1.0, 0.5, 1.2, 0.6, 0.3)),
    new GenericRow(Array[Any](1.5, 1.2, 0.5, 1.4, 0.6, 0.4)))
  val schema = new FrameSchema(List(Column("1", DataTypes.float32),
    Column("2", DataTypes.float32),
    Column("3", DataTypes.float32),
    Column("4", DataTypes.float32),
    Column("5", DataTypes.float32),
    Column("6", DataTypes.float32)))

  "PcaModel" should {

    "create a DAAL PrincipalComponentsModel train and predict" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // model train
      val model = PcaModel.train(frame, List("1", "2", "3", "4", "5", "6"), true, Some(3))
      model shouldBe a[PcaModel]
      assert(model.columnMeans.length == schema.columns.length)

      // predict
      val predictFrame = model.predict(frame, true, true, Some(List("1", "2", "3", "4", "5", "6")), Some(2))
      assert(predictFrame.rowCount() == frame.rowCount())
      assert(predictFrame.schema.hasColumn("p_1"))
      assert(predictFrame.schema.hasColumn("p_2"))
      assert(predictFrame.schema.hasColumn("t_squared_index"))
    }

    "throw an exception for invalid train arguments" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // null frame
      var thrown = the[Exception] thrownBy PcaModel.train(null, List("1", "2", "3", "4", "5", "6"), true, Some(3))
      assert(thrown.getMessage.contains("frame is required"))

      // invalid observation column
      thrown = the[Exception] thrownBy PcaModel.train(frame, List("1", "bogus", "3", "4", "5", "6"), true, Some(3))
      assert(thrown.getMessage.contains("column bogus was not found"))

      // invalid k (greater than number of observation columns
      thrown = the[Exception] thrownBy PcaModel.train(frame, List("1", "bogus", "3", "4", "5", "6"), true, Some(8))
      assert(thrown.getMessage.contains("k must be less than or equal to number of observation columns"))
    }

    "throw an exception for invalid predict arguments" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)
      val model = PcaModel.train(frame, List("1", "2", "3", "4", "5", "6"), true, Some(3))

      // null frame
      var thrown = the[Exception] thrownBy model.predict(null, true, true, Some(List("1", "2", "3", "4", "5", "6")), Some(2))
      assert(thrown.getMessage.contains("frame is required"))

      // invalid observation column
      thrown = the[Exception] thrownBy model.predict(frame, true, true, Some(List("1", "bogus", "3", "4", "5", "6")), Some(2))
      assert(thrown.getMessage.contains("No column named bogus"))

      // invalid c (greater than number of observation columns)
      thrown = the[Exception] thrownBy model.predict(frame, true, true, Some(List("1", "2", "3", "4", "5", "6")), Some(8))
      assert(thrown.getMessage.contains("Number of components must be at most the number of components trained on"))
    }

    "return a prediction when calling score on a trained model" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // model train
      val k = 3
      val model = PcaModel.train(frame, List("1", "2", "3", "4", "5", "6"), true, Some(k))

      // Score
      val inputArray = Array[Any](2.6, 1.7, 0.3, 1.5, 0.8, 0.7)
      assert(model.input().length == inputArray.length)
      val scoreResult = model.score(inputArray)
      assert(scoreResult.length == model.output().length)
      for ((input, i) <- inputArray.zipWithIndex)
        assert(scoreResult(i) == input)
      val pcIndex = inputArray.length
      scoreResult(pcIndex) match {
        case pc: List[_] => assert(pc.length == k)
        case _ => throw new RuntimeException(s"Expected principal components score result to be a List but is ${scoreResult(pcIndex).getClass.getSimpleName}")
      }
    }

  }
}
