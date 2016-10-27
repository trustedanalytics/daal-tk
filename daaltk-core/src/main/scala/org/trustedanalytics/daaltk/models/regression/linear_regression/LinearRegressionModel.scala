package org.trustedanalytics.daaltk.models.regression.linear_regression

import org.apache.commons.lang.StringUtils
import org.apache.spark.SparkContext
import org.json4s.JsonAST.JValue
import org.trustedanalytics.daaltk.DaalUtils
import org.trustedanalytics.daaltk.models.DaalTkModelAdapter
import org.trustedanalytics.sparktk.TkContext
import org.trustedanalytics.sparktk.frame.Frame
import org.trustedanalytics.sparktk.frame.internal.rdd.FrameRdd
import org.trustedanalytics.sparktk.saveload.{ SaveLoad, TkSaveLoad, TkSaveableObject }
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.trustedanalytics.scoring.interfaces.{ ModelMetaData, Field, Model }
import org.trustedanalytics.sparktk.models.ScoringModelUtils
import org.trustedanalytics.daaltk.models.DaalModel
import breeze.linalg.{ DenseVector => BreezeDenseVector }

object LinearRegressionModel extends TkSaveableObject {
  /**
   * Current format version for model save/load
   */
  private val currentFormatVersion: Int = 1

  /**
   * List of format version that are valid for loading
   */
  private val validFormatVersions = List[Int](currentFormatVersion)

  /**
   * ID for the format of how the object is save/load-ed.
   */
  override def formatId: String = "DaalLinearRegressionModel"

  /**
   *
   * @param frame A frame to train or test the model on.
   * @param valueColumn Column name containing the value for each observation.
   * @param observationColumns List of column(s) containing the observations.
   * @param fitIntercept Parameter for whether to fit an intercept term.
   * @return Trained DAAL linear regression model
   */
  def train(frame: Frame,
            valueColumn: String,
            observationColumns: Seq[String],
            fitIntercept: Boolean = true): LinearRegressionModel = {
    require(frame != null, "frame is required")
    require(observationColumns != null && observationColumns.nonEmpty, "observationColumn must not be null nor empty")
    require(StringUtils.isNotEmpty(valueColumn), "valueColumn must not be null nor empty")

    // Validate Daal libraries
    DaalUtils.validateDaalLibraries(DaalUtils.daalDynamicLibraries)

    // Create FrameRdd from the frame
    val frameRdd = new FrameRdd(frame.schema, frame.rdd)

    // Train model
    val trainedModelData = LinearTrainAlgorithm(frameRdd, observationColumns, valueColumn, fitIntercept).train()

    // Compute summary statistics for regression model
    val predictFrameRdd = LinearPredictAlgorithm(trainedModelData.serializedModel, frameRdd, valueColumn, observationColumns).predict()
    val summary = getRegressionMetrics(valueColumn, predictFrameRdd)

    LinearRegressionModel(valueColumn,
      observationColumns,
      trainedModelData.intercept,
      trainedModelData.weights,
      summary.explainedVariance,
      summary.meanAbsoluteError,
      summary.meanSquaredError,
      summary.r2,
      summary.rootMeanSquaredError,
      trainedModelData.serializedModel)
  }

  /**
   * Get regression metrics for trained model
   */
  def getRegressionMetrics(valueColumn: String, predictFrameRdd: FrameRdd): RegressionMetrics = {

    val predictionAndObservations = predictFrameRdd.mapRows(row => {
      val prediction = row.doubleValue(LinearPredictAlgorithm.PredictColumnPrefix + valueColumn)
      val value = row.doubleValue(valueColumn)
      (prediction, value)
    })

    val summary = new RegressionMetrics(predictionAndObservations)
    summary
  }

  /**
   * Create a DAAL LinearRegressionModel, given the specified metadata.
   */
  def loadTkSaveableObject(sc: SparkContext, path: String, formatVersion: Int, tkMetadata: JValue): Any = {
    validateFormatVersion(formatVersion, validFormatVersions: _*)
    val m: LinearRegressionModelTkMetaData = SaveLoad.extractFromJValue[LinearRegressionModelTkMetaData](tkMetadata)

    // Create LinearRegressionModel to return
    LinearRegressionModel(m.valueColumnTrain,
      m.observationColumnsTrain,
      m.intercept,
      m.weights,
      m.explainedVariance,
      m.meanAbsoluteError,
      m.meanSquaredError,
      m.r2,
      m.rootMeanSquaredError,
      m.serializedModel)
  }

  /**
   * Load a DAAL linear regression from the given path
   *
   * @param tc TkContext
   * @param path location
   * @return
   */
  def load(tc: TkContext, path: String): LinearRegressionModel = {
    tc.load(path).asInstanceOf[LinearRegressionModel]
  }
}

/**
 * Intel DAAL Linear Regression Model
 *
 * @param valueColumnTrain Frame's column storing the value of the observation
 * @param observationColumnsTrain Frame's column(s) storing the observations
 * @param intercept The intercept of the trained model
 * @param weights Weights of the trained model
 * @param explainedVariance The explained variance regression score
 * @param meanAbsoluteError The risk function corresponding to the expected value of the absolute error loss or l1-norm loss
 * @param meanSquaredError The risk function corresponding to the expected value of the squared error loss or quadratic loss
 * @param r2 The coefficient of determination of the trained model
 * @param rootMeanSquaredError The square root of the mean squared error
 * @param serializedModel Serialized DAAL linear regression model
 */
case class LinearRegressionModel(valueColumnTrain: String,
                                 observationColumnsTrain: Seq[String],
                                 intercept: Double,
                                 weights: Seq[Double],
                                 explainedVariance: Double,
                                 meanAbsoluteError: Double,
                                 meanSquaredError: Double,
                                 r2: Double,
                                 rootMeanSquaredError: Double,
                                 serializedModel: List[Byte]) extends Serializable with Model with DaalModel {

  /**
   * Array of observation columns used during training
   */
  def observationColumnsAsArray: Array[String] = observationColumnsTrain.toArray

  /**
   * Array of weights of the trained model
   */
  def weightsAsArray: Array[Double] = weights.toArray

  /**
   * Get test metrics for DAAL's Linear Regression with QR decomposition using test frame
   *
   * @param frame The frame to test the linear regression model on
   * @param valueColumn Optional frame column containing the value for the observation
   * @param observationColumns Optional frame column(s) containing the observations
   * @return Test metrics for the Intel DAAL linear regression model
   */
  def test(frame: Frame,
           valueColumn: Option[String] = None,
           observationColumns: Option[Seq[String]] = None): LinearRegressionTestReturn = {
    require(frame != null, "frame is required")
    // Validate DAAL libraries
    DaalUtils.validateDaalLibraries(DaalUtils.daalDynamicLibraries)

    //create RDD from the frame
    val testFrame = new FrameRdd(frame.schema, frame.rdd)
    val testObservationColumns = observationColumns.getOrElse(observationColumnsTrain)
    val testValueColumn = valueColumn.getOrElse(valueColumnTrain)
    require(observationColumnsTrain.length == testObservationColumns.length,
      "Number of observations columns for train and predict should be same")

    val predictFrame = LinearPredictAlgorithm(serializedModel, testFrame, testValueColumn, testObservationColumns).predict()
    val summary = LinearRegressionModel.getRegressionMetrics(testValueColumn, predictFrame)

    LinearRegressionTestReturn(
      summary.explainedVariance,
      summary.meanAbsoluteError,
      summary.meanSquaredError,
      summary.r2,
      summary.rootMeanSquaredError
    )
  }

  /**
   * Predict labels for a test frame using trained Intel DAAL linear regression model.
   *
   * Get predictions for DAAL's Linear Regression with QR decomposition using test frame.
   *
   * @param frame The frame to predict on
   * @param valueColumn Column name containing the value of each observation
   * @param observationColumns List of column(s) containing the observations
   * @return returns predicted frame
   */
  def predict(frame: Frame,
              valueColumn: Option[String] = None,
              observationColumns: Option[Seq[String]] = None): Frame = {
    require(frame != null, "frame is required")
    if (valueColumn.isDefined)
      require(StringUtils.isNotEmpty(valueColumn.get), "valueColumn must not be null nor empty.")
    if (observationColumns.isDefined)
      require(observationColumns.get != null && observationColumns.get.length > 0,
        "observationColumns must not be null nor empty.")

    // Validate DAAL libraries
    DaalUtils.validateDaalLibraries(DaalUtils.daalDynamicLibraries)

    //create RDD from the frame
    val predictFrameRdd = new FrameRdd(frame.schema, frame.rdd)
    val observationColumnsPredict = observationColumns.getOrElse(observationColumnsTrain)
    require(observationColumnsTrain.length == observationColumnsPredict.length,
      "Number of observations columns for train and predict should be same")
    val valueColumnPredict = valueColumn.getOrElse(valueColumnTrain)

    // Predict
    val resultRdd = LinearPredictAlgorithm(serializedModel, predictFrameRdd, valueColumnPredict, observationColumnsPredict).predict()

    // Return predict frame
    new Frame(resultRdd.rdd, resultRdd.schema)
  }

  /**
   * Saves this model to a file
   *
   * @param sc active SparkContext
   * @param path save to path
   */
  override def save(sc: SparkContext, path: String): Unit = {
    val tkMetadata = LinearRegressionModelTkMetaData(valueColumnTrain,
      observationColumnsTrain,
      intercept,
      weights,
      explainedVariance,
      meanAbsoluteError,
      meanSquaredError,
      r2,
      rootMeanSquaredError,
      serializedModel)
    TkSaveLoad.saveTk(sc, path, LinearRegressionModel.formatId, LinearRegressionModel.currentFormatVersion, tkMetadata)
  }

  /**
   * Scores the given row using the trained DAAL Linear Regression model
   *
   * @param row Row of input data
   * @return Input row, plus the score
   */
  override def score(row: Array[Any]): Array[Any] = {
    val breezeWeights = new BreezeDenseVector[Double](weights.toArray)
    val features: Array[Double] = row.map(y => ScoringModelUtils.asDouble(y))
    val breezeFeatures = new BreezeDenseVector[Double](features)
    val prediction = breezeWeights.dot(breezeFeatures) + intercept
    row :+ (prediction)
  }

  /**
   * @return DAAL Linear Regression model metadata
   */
  override def modelMetadata(): ModelMetaData = {
    new ModelMetaData("Intel DAAL Linear Regression Model",
      classOf[LinearRegressionModel].getName,
      classOf[DaalTkModelAdapter].getName,
      Map())
  }

  /**
   * @return fields containing the input names and their data types
   */
  override def input(): Array[Field] = {
    var input = Array[Field]()
    observationColumnsTrain.foreach { name =>
      input = input :+ Field(name, "Double")
    }
    input
  }

  /**
   * @return fields containing the input names and their data types along with the output and its data type
   */
  override def output(): Array[Field] = {
    var output = input()
    output :+ Field("score", "Double")
  }
}

/**
 * TK Metadata that will be stored as part of the model
 *
 * @param valueColumnTrain Frame's column storing the value of the observation
 * @param observationColumnsTrain Frame's column(s) storing the observations
 * @param intercept The intercept of the trained model
 * @param weights Weights of the trained model
 * @param explainedVariance The explained variance regression score
 * @param meanAbsoluteError The risk function corresponding to the expected value of the absolute error loss or l1-norm loss
 * @param meanSquaredError The risk function corresponding to the expected value of the squared error loss or quadratic loss
 * @param r2 The coefficient of determination of the trained model
 * @param rootMeanSquaredError The square root of the mean squared error
 * @param serializedModel Serialized DAAL linear regression model
 */
case class LinearRegressionModelTkMetaData(valueColumnTrain: String,
                                           observationColumnsTrain: Seq[String],
                                           intercept: Double,
                                           weights: Seq[Double],
                                           explainedVariance: Double,
                                           meanAbsoluteError: Double,
                                           meanSquaredError: Double,
                                           r2: Double,
                                           rootMeanSquaredError: Double,
                                           serializedModel: List[Byte]) extends Serializable

/**
 * Results of training DAAL linear regression model
 *
 * @param observationColumns Frame's column(s) storing the observations
 * @param valueColumn Frame's column storing the value of the observation
 * @param intercept The intercept of the trained model
 * @param weights Weights of the trained model
 * @param meanAbsoluteError The risk function corresponding to the expected value of the absolute error loss or l1-norm loss
 * @param meanSquaredError The risk function corresponding to the expected value of the squared error loss or quadratic loss
 * @param r2 The coefficient of determination of the trained model
 * @param rootMeanSquaredError The square root of the mean squared error
 */
case class LinearRegressionTrainReturn(observationColumns: Seq[String],
                                       valueColumn: String,
                                       intercept: Double,
                                       weights: Seq[Double],
                                       explainedVariance: Double,
                                       meanAbsoluteError: Double,
                                       meanSquaredError: Double,
                                       r2: Double,
                                       rootMeanSquaredError: Double) {
  require(observationColumns != null && observationColumns.nonEmpty, "observationColumn must not be null nor empty")
  require(valueColumn != null && !valueColumn.isEmpty, "valueColumn must not be null nor empty")
  require(weights != null, "model weights must not be null")
}

/**
 * Return of Linear Regression test plugin
 * @param explainedVariance The explained variance regression score
 * @param meanAbsoluteError The risk function corresponding to the expected value of the absolute error loss or l1-norm loss
 * @param meanSquaredError The risk function corresponding to the expected value of the squared error loss or quadratic loss
 * @param r2 The coefficient of determination
 * @param rootMeanSquaredError The square root of the mean squared error
 */
case class LinearRegressionTestReturn(explainedVariance: Double,
                                      meanAbsoluteError: Double,
                                      meanSquaredError: Double,
                                      r2: Double,
                                      rootMeanSquaredError: Double)