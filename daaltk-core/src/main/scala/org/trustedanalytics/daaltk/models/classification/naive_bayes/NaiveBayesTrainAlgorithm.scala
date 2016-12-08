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

import com.intel.daal.algorithms.ModelSerializer
import com.intel.daal.algorithms.classifier.training.{ InputId, TrainingDistributedInputId, TrainingResultId }
import com.intel.daal.algorithms.multinomial_naive_bayes.Model
import com.intel.daal.algorithms.multinomial_naive_bayes.training._
import com.intel.daal.data_management.data.HomogenNumericTable
import com.intel.daal.services.DaalContext
import org.trustedanalytics.sparktk.frame.internal.rdd.FrameRdd
import org.apache.spark.rdd.RDD
import org.trustedanalytics.daaltk.DistributedAlgorithm
import org.trustedanalytics.daaltk.models.tables.DistributedLabeledTable
import org.trustedanalytics.daaltk.models.tables.DaalConversionImplicits._

/**
 * Train Intel DAAL naive bayes model using QR decomposition
 *
 * @param frameRdd Input frame
 * @param observationColumns Feature columns
 * @param labelColumn Label column
 * @param numClasses Number of classes
 * @param lambda Imagined occurrences of features
 */
case class DaalNaiveBayesTrainAlgorithm(frameRdd: FrameRdd,
                                        observationColumns: Seq[String],
                                        labelColumn: String,
                                        numClasses: Int,
                                        lambda: Double = 1d,
                                        classPrior: Option[Array[Double]] = None) extends DistributedAlgorithm[TrainingPartialResult, TrainingResult] {
  private val trainTables = DistributedLabeledTable.createTable(frameRdd, observationColumns, List(labelColumn))

  /**
   * Train Intel DAAL naive bayes model using QR decomposition
   * @return Trained naive bayes model
   */
  def train(): DaalNaiveBayesModelData = {
    val context = new DaalContext
    var serializedModel: List[Byte] = null
    var trainedModel: Model = null
    var classLogPrior: Array[Double] = null
    var featureLogProb: Array[Array[Double]] = null

    try {
      val partialResultsRdd = computePartialResults()
      val trainingResult = mergePartialResults(context, partialResultsRdd)
      trainedModel = trainingResult.get(TrainingResultId.model)
      serializedModel = serializeTrainedModel(trainedModel)
      classLogPrior = trainedModel.getLogP().toDoubleArray()
      featureLogProb = trainedModel.getLogTheta().toArrayOfDoubleArray()
    }
    catch {
      case ex: Exception => throw new RuntimeException("Could not train model:", ex)
    }
    finally {
      context.dispose()
    }
    DaalNaiveBayesModelData(serializedModel, classLogPrior, featureLogProb, classPrior)
  }

  /**
   * Compute partial results for multinomial Naive Bayes model
   *
   * @return Partial result of training
   */
  override def computePartialResults(): RDD[TrainingPartialResult] = {

    val partialResultRdd = trainTables.rdd.map(table => {
      val context = new DaalContext()
      var partialResult: TrainingPartialResult = null

      try {
        val featureTable = table.features
        val labelTable = table.labels

        val naiveBayesTraining = new TrainingDistributedStep1Local(context, classOf[java.lang.Double],
          TrainingMethod.defaultDense, numClasses)
        naiveBayesTraining.input.set(InputId.data, featureTable.getUnpackedTable(context))
        naiveBayesTraining.input.set(InputId.labels, labelTable.getUnpackedTable(context))

        val alphaParameter = DaalNaiveBayesParameters.getAlphaParameter(context, lambda, observationColumns.length)
        naiveBayesTraining.parameter.setAlpha(alphaParameter)
        if (classPrior.isDefined) {
          val priorParameter = DaalNaiveBayesParameters.getClassPriorParameter(context, classPrior.get)
          naiveBayesTraining.parameter.setPriorClassEstimates(priorParameter)
        }

        partialResult = naiveBayesTraining.compute()
        partialResult.pack()
      }
      catch {
        case ex: Exception => throw new RuntimeException("Could not compute partial model:", ex)
      }
      finally {
        context.dispose()
      }
      partialResult
    })
    partialResultRdd
  }

  /**
   * Merge partial results to generate final training result that contains naive bayes model
   *
   * @param context DAAL Context
   * @param rdd RDD of partial results
   * @return Final result of algorithm
   */
  override def mergePartialResults(context: DaalContext, rdd: RDD[TrainingPartialResult]): TrainingResult = {
    val naiveBayesTraining = new TrainingDistributedStep2Master(context, classOf[java.lang.Double],
      TrainingMethod.defaultDense, numClasses)

    /* Build and retrieve final naive bayes model */
    val partialModelArray = rdd.collect()
    partialModelArray.foreach { partialModel =>
      partialModel.unpack(context)
      naiveBayesTraining.input.add(TrainingDistributedInputId.partialModels, partialModel)
    }

    naiveBayesTraining.compute()
    val trainingResult = naiveBayesTraining.finalizeCompute()
    trainingResult
  }

  /**
   * Serialize trained model to byte array
   * @param trainedModel Trained model
   * @return Serialized model
   */
  private def serializeTrainedModel(trainedModel: Model): List[Byte] = {
    ModelSerializer.serializeNaiveBayesModel(trainedModel).toList
  }

}

/**
 * DAAL Naive Bayes trained model data
 *
 * @param serializedModel Serialized Naive Bayes model
 * @param classLogPrior Smoothed empirical log probability for each class.
 * @param featureLogProb Empirical log probability of features given a class, P(x_i|y).
 * @param classPrior Optional prior probabilities of classes
 */
case class DaalNaiveBayesModelData(serializedModel: List[Byte],
                                   classLogPrior: Array[Double],
                                   featureLogProb: Array[Array[Double]],
                                   classPrior: Option[Array[Double]] = None)

/**
 * Helper methods for getting DAAL Naive Bayes parameters
 */
object DaalNaiveBayesParameters {

  /**
   * Create DAAL numeric table with additive smoothing parameter
   *
   * @param context DAAL context
   * @param alpha Additive smoothing parameter
   * @param featureLength Feature length
   * @return Numeric table with additive smooting parameter
   */
  def getAlphaParameter(context: DaalContext, alpha: Double, featureLength: Int): HomogenNumericTable = {
    val alphaParameters = Array.fill[Double](featureLength)(alpha)
    new HomogenNumericTable(context, alphaParameters, alphaParameters.length, 1L)
  }

  /**
   * Create numeric table with class priors
   *
   * @param context DAAL context
   * @param classPrior Class priors
   * @return Numeric table with class priors
   */
  def getClassPriorParameter(context: DaalContext, classPrior: Array[Double]): HomogenNumericTable = {
    new HomogenNumericTable(context, classPrior, classPrior.length, 1L)
  }
}

