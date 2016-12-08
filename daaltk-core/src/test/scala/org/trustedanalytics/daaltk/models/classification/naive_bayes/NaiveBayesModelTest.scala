/**
 * /*
 * // Copyright (c) 2016 Intel Corporation 
 * //
 * // Licensed under the Apache License, Version 2.0 (the "License");
 * // you may not use this file except in compliance with the License.
 * // You may obtain a copy of the License at
 * //
 * //      http://www.apache.org/licenses/LICENSE-2.0
 * //
 * // Unless required by applicable law or agreed to in writing, software
 * // distributed under the License is distributed on an "AS IS" BASIS,
 * // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * // See the License for the specific language governing permissions and
 * // limitations under the License.
 * */
 */
package org.trustedanalytics.daaltk.models.classification.naive_bayes

import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.Row
import org.scalatest.Matchers
import org.trustedanalytics.daaltk.testutils.TestingSparkContextWordSpec
import org.trustedanalytics.sparktk.frame.{ FrameSchema, Frame, DataTypes, Column }

class NaiveBayesModelTest extends TestingSparkContextWordSpec with Matchers {

  val frameData: Array[Row] = Array(
    new GenericRow(Array[Any](1, 19.8446136104, 2.2985856384)),
    new GenericRow(Array[Any](1, 16.8973559126, 2.6933495054)),
    new GenericRow(Array[Any](1, 5.5548729596, 2.7777687995)),
    new GenericRow(Array[Any](0, 46.1810010826, 3.1611961917)),
    new GenericRow(Array[Any](0, 44.3117586448, 3.3458963222)),
    new GenericRow(Array[Any](0, 34.6334526911, 3.6429838715)))
  val schema = new FrameSchema(List(Column("Class", DataTypes.int32),
    Column("Dim_1", DataTypes.float32),
    Column("Dim_2", DataTypes.float32)))

  "NaiveBayesModel" should {

    "create a DAAL NaiveBayesModel train, test, and predict" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // Model training
      val model = NaiveBayesModel.train(frame, "Class", List("Dim_1", "Dim_2"), 2)
      model shouldBe a[NaiveBayesModel]

      // Check the trained model's parameters
      assert(model.classLogPrior.length == 2)
      assertAlmostEqual(model.classLogPrior(0), -0.6931471805599453)
      assertAlmostEqual(model.classLogPrior(1), -0.6931471805599453)
      assert(model.featureLogProb.length == 2)
      assert(model.featureLogProb(0).length == 2)
      assert(model.featureLogProb(1).length == 2)
      assertAlmostEqual(model.featureLogProb(0)(0), -0.07696104113612832)
      assertAlmostEqual(model.featureLogProb(0)(1), -2.6026896854443837)
      assertAlmostEqual(model.featureLogProb(1)(0), -0.15762894420358317)
      assertAlmostEqual(model.featureLogProb(1)(1), -1.9252908618525777)
      assert(model.lambdaParameter == 1.0)

      // Model test
      val testOutput = model.test(frame, "Class")
      assertAlmostEqual(testOutput.accuracy, 0.833333333333)
      assert(testOutput.fMeasure == 0.8)
      assert(testOutput.precision == 1.0)
      assertAlmostEqual(testOutput.recall, 0.666666666667)
      assert(testOutput.confusionMatrix.numColumns == 2)
      assert(testOutput.confusionMatrix.numRows == 2)
      val confusionMatrix = testOutput.confusionMatrix.getMatrix
      assert(confusionMatrix(0).sameElements(Array(2, 1)))
      assert(confusionMatrix(1).sameElements(Array(0, 3)))

      // Model predict
      val predictedFrame = model.predict(frame)
      assert(predictedFrame.schema.hasColumn("predicted_class") == true)
      val predictIndex = predictedFrame.schema.columnIndex("predicted_class")
      assert(predictedFrame.rowCount() == 6)
      val predictedFrameData = predictedFrame.take(6)
      val predictValues = predictedFrameData.map(row => row.getDouble(predictIndex))
      assert(predictValues.sameElements(Array(0.0, 1.0, 1.0, 0.0, 0.0, 0.0)))
    }

    "throw exceptions for invalid train parameters" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // invalid label column
      var thrown = the[Exception] thrownBy NaiveBayesModel.train(frame, "bogus", List("Dim_1", "Dim_2"), 2)
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))

      // invalid observation column
      thrown = the[Exception] thrownBy NaiveBayesModel.train(frame, "Class", List("Dim_1", "bogus"), 2)
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))

      // invalid num classes
      thrown = the[Exception] thrownBy NaiveBayesModel.train(frame, "Class", List("Dim_1", "Dim_2"), 0)
      assert(thrown.getMessage.contains("number of classes must be greater than 1"))
    }

    "throw exceptions for invalid test parameters" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)
      val model = NaiveBayesModel.train(frame, "Class", List("Dim_1", "Dim_2"), 2)

      // null frame
      var thrown = the[Exception] thrownBy model.test(null, "Class")
      assert(thrown.getMessage.contains("frame is required"))

      // invalid label column
      thrown = the[Exception] thrownBy model.test(frame, "bogus")
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))

      // invalid observation column
      thrown = the[Exception] thrownBy model.test(frame, "Class", Some(List("bogus", "Dim_2")))
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))
    }

    "throw exceptions for invalid predict parameters" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)
      val model = NaiveBayesModel.train(frame, "Class", List("Dim_1", "Dim_2"), 2)

      // null frame
      var thrown = the[Exception] thrownBy model.predict(null)
      assert(thrown.getMessage.contains("frame is required"))

      // invalid observation column
      thrown = the[Exception] thrownBy model.predict(frame, Some(List("bogus", "Dim_2")))
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))
    }

    "return a prediction when calling score on a trained model" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // Model training
      val model = NaiveBayesModel.train(frame, "Class", List("Dim_1", "Dim_2"), 2)

      // Score
      val dim1 = 19.8446136104
      val dim2 = 2.2985856384
      val inputArray = Array[Any](dim1, dim2)
      assert(model.input().length == inputArray.length)
      val scoreResult = model.score(inputArray)
      val expectedPrediction = 0.0
      assert(scoreResult.length == model.output().length)
      assert(scoreResult(0) == dim1)
      assert(scoreResult(1) == dim2)
      scoreResult(2) match {
        case prediction: Double => assert(prediction == expectedPrediction)
        case _ => throw new RuntimeException(s"Expected prediction to be a Double but is ${scoreResult(2).getClass.getSimpleName}")
      }
    }
  }

}
