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
package org.trustedanalytics.daaltk.models.classification.naive_bayes

import com.intel.daal.algorithms.ModelSerializer
import org.apache.commons.lang.StringUtils
import org.apache.spark.SparkContext
import org.trustedanalytics.daaltk.models.{ DaalModel, DaalTkModelAdapter }
import org.trustedanalytics.sparktk.TkContext
import org.trustedanalytics.sparktk.frame._
import org.trustedanalytics.sparktk.frame.internal.RowWrapper
import org.trustedanalytics.sparktk.frame.internal.rdd.{ RowWrapperFunctions, FrameRdd, ScoreAndLabel }
import org.trustedanalytics.sparktk.models.ScoringModelUtils
import org.trustedanalytics.sparktk.saveload.{ SaveLoad, TkSaveLoad, TkSaveableObject }
import org.trustedanalytics.sparktk.frame.internal.ops.classificationmetrics.{ ClassificationMetricsFunctions, ClassificationMetricValue }
import org.trustedanalytics.scoring.interfaces.{ ModelMetaData, Field, Model }
import scala.language.implicitConversions
import org.json4s.JsonAST.JValue
import java.lang
import com.intel.daal.services.DaalContext
import com.intel.daal.algorithms.classifier.prediction.{ ModelInputId, NumericTableInputId, PredictionResultId }
import com.intel.daal.algorithms.multinomial_naive_bayes.prediction.{ PredictionMethod, PredictionBatch }
import com.intel.daal.data_management.data.HomogenNumericTable

object NaiveBayesModel extends TkSaveableObject {
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
  override def formatId: String = "DaalNaiveBayesModel"

  /**
   * Build a multinomial DAAL Naive Bayes model.
   *
   * @param frame The frame containing the data to train on
   * @param labelColumn Column containing the label for each observation
   * @param observationColumns Column(s) containing the observations
   * @param numClasses Number of classes.
   */
  def train(frame: Frame,
            labelColumn: String,
            observationColumns: Seq[String],
            numClasses: Int = 2): NaiveBayesModel = {
    require(frame != null, "frame is required")
    require(observationColumns != null && observationColumns.nonEmpty, "observation column must not be null nor empty")
    require(StringUtils.isNotEmpty(labelColumn), "label column must not be null nor empty")
    require(numClasses > 1, "number of classes must be greater than 1")

    val trainFrame = new FrameRdd(frame.schema, frame.rdd)

    //TODO: Convert to plugin argument once fix for setting DAAL naive Bayes parameters is available
    val lambdaParameter = 1d
    val naiveBayesModelData = new DaalNaiveBayesTrainAlgorithm(
      trainFrame,
      observationColumns,
      labelColumn,
      numClasses,
      lambdaParameter
    ).train()

    // Return trained model and training data/info
    NaiveBayesModel(naiveBayesModelData.serializedModel,
      labelColumn,
      observationColumns,
      lambdaParameter,
      numClasses,
      naiveBayesModelData.classLogPrior,
      naiveBayesModelData.featureLogProb,
      naiveBayesModelData.classPrior)
  }

  /**
   * Load a NaiveBayesModel from the given path
   */
  def load(tc: TkContext, path: String): NaiveBayesModel = {
    tc.load(path).asInstanceOf[NaiveBayesModel]
  }

  /**
   * Create a NaiveBayesModel using the specified metadata
   */
  def loadTkSaveableObject(sc: SparkContext, path: String, formatVersion: Int, tkMetadata: JValue): Any = {
    validateFormatVersion(formatVersion, validFormatVersions: _*)
    val m: NaiveBayesModelTkMetaData = SaveLoad.extractFromJValue[NaiveBayesModelTkMetaData](tkMetadata)

    // Create NaiveBayesModel to return
    NaiveBayesModel(m.serializedModel,
      m.labelColumn,
      m.trainingObservationColumns,
      m.lambdaParameter,
      m.numClasses,
      m.classLogPrior,
      m.featureLogProb,
      m.classPrior)
  }
}

/**
 * Trained DAAL Naive Bayes Model
 *
 * @param serializedModel Serialized DAAL Naive Bayes model
 * @param labelColumn Label column for trained model
 * @param trainingObservationColumns Handle to the observation columns of the data frame
 * @param lambdaParameter Smoothing parameter used during model training
 * @param numClasses Number of classes
 * @param classLogPrior Smoothed empirical log probability for each class.
 * @param featureLogProb Empirical log probability of features given a class, P(x_i|y).
 * @param classPrior Optional prior probabilities of classes
 */
case class NaiveBayesModel private[naive_bayes] (serializedModel: List[Byte],
                                                 labelColumn: String,
                                                 trainingObservationColumns: Seq[String],
                                                 lambdaParameter: Double,
                                                 numClasses: Int,
                                                 classLogPrior: Array[Double],
                                                 featureLogProb: Array[Array[Double]],
                                                 classPrior: Option[Array[Double]] = None) extends Serializable with Model with DaalModel {

  implicit def rowWrapperToRowWrapperFunctions(rowWrapper: RowWrapper): RowWrapperFunctions = {
    new RowWrapperFunctions(rowWrapper)
  }

  /**
   * Predict labels for data points using trained multinomial Naive Bayes model, and create a new frame
   * revision with existing columns and a new predicted label's column.
   *
   * @param frame - frame to add predictions to
   * @param observationColumns Column(s) containing the observations whose labels are to be predicted. By default,
   *                           we predict the labels over columns the NaiveBayesModel was trained on.
   */
  def predict(frame: Frame, observationColumns: Option[List[String]] = None): Frame = {

    require(frame != null, "frame is required")
    if (observationColumns.isDefined) {
      require(trainingObservationColumns.length == observationColumns.get.length,
        "Number of columns for train and predict should be same")
    }

    val frameRdd = new FrameRdd(frame.schema, frame.rdd)

    // predicting a label for the observation columns
    val naiveBayesColumns = observationColumns.getOrElse(trainingObservationColumns)
    val predictColumn = "predicted_class"
    val predictFrame = new NaiveBayesPredictAlgorithm(serializedModel, lambdaParameter, numClasses, classPrior,
      frameRdd, naiveBayesColumns, predictColumn).predict()

    new Frame(predictFrame.rdd, predictFrame.schema)
  }

  /**
   * Predict label using Intel DAAL naive bayes model
   * @param features Array with input features
   * @return score
   */
  private def scoreRow(features: Array[Double]): Double = {
    val context = new DaalContext()
    var prediction: Double = Double.NaN

    try {
      val predictAlgorithm = new PredictionBatch(context, classOf[lang.Double],
        PredictionMethod.defaultDense, numClasses)
      val testTable = new HomogenNumericTable(context, features, features.length, 1L)
      val trainedModel = ModelSerializer.deserializeNaiveBayesModel(context, serializedModel.toArray)
      predictAlgorithm.input.set(NumericTableInputId.data, testTable)
      predictAlgorithm.input.set(ModelInputId.model, trainedModel)

      val alphaParameter = DaalNaiveBayesParameters.getAlphaParameter(context,
        lambdaParameter, trainingObservationColumns.length)
      predictAlgorithm.parameter.setAlpha(alphaParameter)
      if (classPrior.isDefined) {
        val priorParameter = DaalNaiveBayesParameters.getClassPriorParameter(context, classPrior.get)
        predictAlgorithm.parameter.setPriorClassEstimates(priorParameter)
      }

      /* Compute and retrieve prediction results */
      val partialResult = predictAlgorithm.compute()

      val predictions = partialResult.get(PredictionResultId.prediction).asInstanceOf[HomogenNumericTable]
      prediction = predictions.getDoubleArray.head
    }
    catch {
      case ex: Exception => throw new RuntimeException("Could not score model:", ex)
    }
    finally {
      context.dispose()
    }
    prediction
  }

  /**
   * Predict the labels for a test frame and run classification metrics on predicted and target labels.
   *
   * @param frame Frame to test the NaiveBayes model
   * @param labelColumn containing the actual label for each observation.
   * @param observationColumns Column(s) containing the observations whose labels are to be predicted.
   *                           By default, we predict the labels over columns the NaiveBayesModel
   * @return A dictionary with classification metrics. The data returned is composed of the following keys\:
   *         'accuracy' : double  The proportion of predictions that are correctly identified
   *         'confusion_matrix' : dictionary  A table used to describe the performance of a classification model
   *         'f_measure' : double  The harmonic mean of precision and recall
   *         'precision' : double  The proportion of predicted positive instances that are correctly identified
   *         'recall' : double  The proportion of positive instances that are correctly identified.
   */
  def test(frame: Frame, labelColumn: String, observationColumns: Option[List[String]] = None): ClassificationMetricValue = {
    require(frame != null, "frame is required")

    if (observationColumns.isDefined) {
      require(trainingObservationColumns.length == observationColumns.get.length,
        "Number of columns for train and predict should be same")
    }

    val frameRdd = new FrameRdd(frame.schema, frame.rdd)

    //predicting a label for the observation columns
    val testObservationColumns = observationColumns.getOrElse(trainingObservationColumns)
    val predictColumn = "predicted_" + labelColumn
    val predictFrame = new NaiveBayesPredictAlgorithm(serializedModel, lambdaParameter, numClasses, classPrior, frameRdd,
      testObservationColumns, predictColumn).predict()

    //predicting and testing
    val scoreAndLabelRdd = predictFrame.toScoreAndLabelRdd(row => {
      val labeledPoint = row.valuesAsLabeledPoint(testObservationColumns, labelColumn)
      val score = row.doubleValue(predictColumn)
      ScoreAndLabel(score, labeledPoint.label)
    })

    //Run classification metrics
    numClasses match {
      case 2 => {
        val posLabel: Double = 1.0d
        ClassificationMetricsFunctions.binaryClassificationMetrics(scoreAndLabelRdd, posLabel)
      }
      case _ => ClassificationMetricsFunctions.multiclassClassificationMetrics(scoreAndLabelRdd)
    }
  }

  /**
   * Saves this model to a file
   * @param sc active SparkContext
   * @param path save to path
   */
  override def save(sc: SparkContext, path: String): Unit = {
    val tkMetadata = NaiveBayesModelTkMetaData(serializedModel, labelColumn, trainingObservationColumns, lambdaParameter,
      numClasses, classLogPrior, featureLogProb, classPrior)
    TkSaveLoad.saveTk(sc, path, NaiveBayesModel.formatId, NaiveBayesModel.currentFormatVersion, tkMetadata)
  }

  /**
   * Scores the given row using the trained DAAL Naive Bayes model
   * @param row Row of input data
   * @return Input row, plus the score
   */
  override def score(row: Array[Any]): Array[Any] = {
    val features: Array[Double] = row.map(y => ScoringModelUtils.asDouble(y))
    val prediction = scoreRow(features)

    row :+ (prediction)
  }

  /**
   * @return DAAL Naive Bayes model metadata
   */
  override def modelMetadata(): ModelMetaData = {
    new ModelMetaData("Intel DAAL Naive Bayes Model", classOf[NaiveBayesModel].getName, classOf[DaalTkModelAdapter].getName, Map())
  }

  /**
   * @return fields containing the input names and their data types
   */
  override def input(): Array[Field] = {
    trainingObservationColumns.map(name => Field(name, "Double")).toArray
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
 * @param serializedModel Serialized DAAL Naive Bayes model
 * @param labelColumn Label column for trained model
 * @param trainingObservationColumns Handle to the observation columns of the data frame
 * @param lambdaParameter Smoothing parameter used during model training
 * @param numClasses Number of classes
 * @param classLogPrior Smoothed empirical log probability for each class.
 * @param featureLogProb Empirical log probability of features given a class, P(x_i|y).
 * @param classPrior Optional prior probabilities of classes
 */
case class NaiveBayesModelTkMetaData(serializedModel: List[Byte],
                                     labelColumn: String,
                                     trainingObservationColumns: Seq[String],
                                     lambdaParameter: Double,
                                     numClasses: Int,
                                     classLogPrior: Array[Double],
                                     featureLogProb: Array[Array[Double]],
                                     classPrior: Option[Array[Double]]) extends Serializable

/**
 * Return for Intel DAAL Naive Bayes train plugin
 * @param classLogPrior Smoothed empirical log probability for each class.
 * @param featureLogProb Empirical log probability of features given a class, P(x_i|y).
 */
case class NaiveBayesTrainReturn(classLogPrior: Array[Double], featureLogProb: Array[Array[Double]])