package org.trustedanalytics.daaltk.models.classification.naive_bayes

import org.apache.commons.lang.StringUtils
import org.apache.spark.SparkContext
import org.trustedanalytics.sparktk.TkContext
import org.trustedanalytics.sparktk.frame._
import org.trustedanalytics.sparktk.frame.internal.RowWrapper
import org.trustedanalytics.sparktk.frame.internal.rdd.{ RowWrapperFunctions, FrameRdd, ScoreAndLabel }
import org.trustedanalytics.sparktk.saveload.{ SaveLoad, TkSaveLoad, TkSaveableObject }
import com.intel.daal.algorithms.multinomial_naive_bayes.Model
import org.trustedanalytics.sparktk.frame.internal.ops.classificationmetrics.{ ClassificationMetricsFunctions, ClassificationMetricValue }

import scala.language.implicitConversions
import org.json4s.JsonAST.JValue

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
                                                 classPrior: Option[Array[Double]] = None) extends Serializable {

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
  def save(sc: SparkContext, path: String): Unit = {
    val tkMetadata = NaiveBayesModelTkMetaData(serializedModel, labelColumn, trainingObservationColumns, lambdaParameter,
      numClasses, classLogPrior, featureLogProb, classPrior)
    TkSaveLoad.saveTk(sc, path, NaiveBayesModel.formatId, NaiveBayesModel.currentFormatVersion, tkMetadata)
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