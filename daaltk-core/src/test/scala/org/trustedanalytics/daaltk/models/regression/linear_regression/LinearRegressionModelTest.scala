package org.trustedanalytics.daaltk.models.regression.linear_regression

import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.apache.spark.sql.Row
import org.scalatest.Matchers
import org.trustedanalytics.daaltk.testutils.TestingSparkContextWordSpec
import org.trustedanalytics.sparktk.frame.{ FrameSchema, Frame, DataTypes, Column }

class LinearRegressionModelTest extends TestingSparkContextWordSpec with Matchers {

  val frameData: Array[Row] = Array(
    new GenericRow(Array[Any](0, 0)),
    new GenericRow(Array[Any](1, 2.5)),
    new GenericRow(Array[Any](2, 5.0)),
    new GenericRow(Array[Any](3, 7.5)),
    new GenericRow(Array[Any](4, 10)),
    new GenericRow(Array[Any](5, 12.5)),
    new GenericRow(Array[Any](6, 13.0)),
    new GenericRow(Array[Any](7, 17.15)),
    new GenericRow(Array[Any](8, 18.5)),
    new GenericRow(Array[Any](9, 23.5)))
  val schema = new FrameSchema(List(Column("x1", DataTypes.float32), Column("y", DataTypes.string)))

  "LinearRegressionModel" should {

    "create a DAAL LinearRegressionModel train, test and predict" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // Model training
      val model = LinearRegressionModel.train(frame, "y", List("x1"))
      model shouldBe a[LinearRegressionModel]

      // Check properties
      assertAlmostEqual(model.explainedVariance, 49.2759280303)
      assertAlmostEqual(model.intercept, -0.0327272727273)
      assertAlmostEqual(model.meanAbsoluteError, 0.529939393939)
      assertAlmostEqual(model.meanSquaredError, 0.630096969697)
      assertAlmostEqual(model.r2, 0.987374330661)
      assertAlmostEqual(model.rootMeanSquaredError, 0.793786476136)
      assert(model.weightsAsArray.length == 1)
      assertAlmostEqual(model.weightsAsArray(0), 2.443939393939393)

      // Model test
      val testOutput = model.test(frame, Some("y"), None)

      // Check properties
      assertAlmostEqual(testOutput.explainedVariance, 49.2759280303)
      assertAlmostEqual(testOutput.meanAbsoluteError, 0.529939393939)
      assertAlmostEqual(testOutput.meanSquaredError, 0.630096969697)
      assertAlmostEqual(testOutput.r2, 0.987374330661)
      assertAlmostEqual(testOutput.rootMeanSquaredError, 0.793786476136)

      // Predict
      val predictFrame = model.predict(frame, None, Some(List("x1")))
      predictFrame shouldBe a[Frame]
      assert(predictFrame.schema.hasColumn("predict_y"))
    }

    "throw an exception for invalid train arguments" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // null frame
      var thrown = the[Exception] thrownBy LinearRegressionModel.train(null, "y", List("x1"))
      assert(thrown.getMessage.contains("frame is required"))

      // invalid value column
      thrown = the[Exception] thrownBy LinearRegressionModel.train(frame, "bogus", List("x1"))
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))

      // invalid observation column
      thrown = the[Exception] thrownBy LinearRegressionModel.train(frame, "y", List("bogus"))
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))
    }

    "throw an exception for invalid test arguments" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)
      val model = LinearRegressionModel.train(frame, "y", List("x1"))

      // frame is required
      var thrown = the[Exception] thrownBy model.test(null, Some("y"), None)
      assert(thrown.getMessage.contains("frame is required"))

      // invalid value column
      thrown = the[Exception] thrownBy model.test(frame, Some("bogus"))
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))

      // invalid observation column
      thrown = the[Exception] thrownBy model.test(frame, Some("y"), Some(List("bogus")))
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))
    }

    "throw an exception for invalid predict arguments" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)
      val model = LinearRegressionModel.train(frame, "y", List("x1"))

      // frame is required
      var thrown = the[Exception] thrownBy model.predict(null, None, None)
      assert(thrown.getMessage.contains("frame is required"))

      // invalid observation column
      thrown = the[Exception] thrownBy model.predict(frame, None, Some(List("bogus")))
      assert(thrown.getMessage.contains("Invalid column name bogus provided"))
    }

    "return a prediction when calling score on a trained model" in {
      val rdd = sparkContext.parallelize(frameData)
      val frame = new Frame(rdd, schema)

      // Model training
      val model = LinearRegressionModel.train(frame, "y", List("x1"))

      // Score
      val inputData = 5.0
      val inputArray = Array[Any](inputData)
      assert(model.input().length == inputArray.length)
      val scoreResult = model.score(inputArray)
      assert(scoreResult.length == model.output().length)
      assert(scoreResult(0) == inputData)
      val expectedScore = 12.186969696969696
      scoreResult(1) match {
        case score: Double => assertAlmostEqual(score, expectedScore)
        case _ => throw new RuntimeException(s"Expected prediction to be an Double but is ${scoreResult(1).getClass.getSimpleName}")
      }
    }
  }
}
