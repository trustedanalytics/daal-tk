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
package org.trustedanalytics.daaltk.models.clustering.kmeans

import java.lang
import com.intel.daal.algorithms.kmeans.{ ResultId, InputId, Method, Batch }
import com.intel.daal.data_management.data.HomogenNumericTable
import com.intel.daal.services.DaalContext
import org.apache.spark.SparkContext
import org.apache.commons.lang.StringUtils
import org.trustedanalytics.daaltk.DaalUtils
import org.trustedanalytics.daaltk.models.{ DaalModel, DaalTkScoringModelUtils, DaalTkModelAdapter }
import org.trustedanalytics.sparktk.TkContext
import org.trustedanalytics.sparktk.frame.internal.rdd.FrameRdd
import org.trustedanalytics.sparktk.frame._
import org.trustedanalytics.sparktk.saveload.{ SaveLoad, TkSaveLoad, TkSaveableObject }
import scala.language.implicitConversions
import org.json4s.JsonAST.JValue
import org.trustedanalytics.scoring.interfaces.{ ModelMetaData, Field, Model }
import org.trustedanalytics.sparktk.models.ScoringModelUtils

object KMeansModel extends TkSaveableObject {
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
  override def formatId: String = "DaalKMeansModel"

  /**
   * @param frame The frame containing the data to train on
   * @param observationColumns The columns containing the observations
   * @param columnScalings Optional column scalings for each of the observation columns. The scaling value is
   *                       multiplied by the corresponding value in the observation column.
   * @param k Desired number of clusters.
   * @param maxIterations Number of iterations for which the algorithm should run.
   * @param labelColumn Optional name of output column with index of cluster each observation belongs to.
   */
  def train(frame: Frame,
            observationColumns: Seq[String],
            columnScalings: Option[List[Double]] = None,
            k: Int = 2,
            maxIterations: Int = 100,
            labelColumn: String = "predicted_cluster"): KMeansModel = {
    require(frame != null, "frame is required")
    require(observationColumns != null && observationColumns.nonEmpty, "observation columns must not be null nor empty")
    require(columnScalings != null && (columnScalings.isEmpty || observationColumns.length == columnScalings.get.length),
      "column scalings must be empty or the same size as observation columns")
    require(StringUtils.isNotEmpty(labelColumn), "label column must not be null nor empty")
    require(k > 0, "k must be at least 1")
    require(maxIterations > 0, "max iterations must be a positive value")

    // Validate DAAL libraries
    DaalUtils.validateDaalLibraries(DaalUtils.daalDynamicLibraries)

    val frameRdd = new FrameRdd(frame.schema, frame.rdd)

    // Train model
    val results = KMeansFunctions.trainKMeansModel(frameRdd, observationColumns, columnScalings, k, maxIterations, labelColumn)
    val centroids = for (i <- 0 until k) yield results.centroids("Cluster:" + i.toString)

    KMeansModel(observationColumns,
      labelColumn,
      k,
      centroids.toArray,
      results.clusterSize.keys.toArray,
      results.clusterSize.values.toArray,
      columnScalings)
  }

  /**
   * Load a KMeansModel from the given path
   * @param tc TkContext
   * @param path location
   * @return
   */
  def load(tc: TkContext, path: String): KMeansModel = {
    tc.load(path).asInstanceOf[KMeansModel]
  }

  /**
   * Create a DAAL KMeans Model give the specified metadata
   */
  def loadTkSaveableObject(sc: SparkContext, path: String, formatVersion: Int, tkMetadata: JValue): Any = {
    validateFormatVersion(formatVersion, validFormatVersions: _*)
    val m: KMeansModelTkMetaData = SaveLoad.extractFromJValue[KMeansModelTkMetaData](tkMetadata)

    // Create KMeansModel to return
    KMeansModel(m.trainingObservationColumns,
      m.trainingLabelColumn,
      m.k,
      m.centroids,
      m.clusters,
      m.clusterSize,
      m.columnScalings)
  }
}

/**
 * Trained DAAL KMeansModel
 *
 * @param trainingObservationColumns  Column(s) containing the observations whose clusters are to be predicted.
 * @param trainingLabelColumn Name of output column with index of cluster each observation belongs to.
 * @param k Desired number of clusters
 * @param centroids Cluster centroids
 * @param clusters Cluster labels
 * @param clusterSize Cluster sizes
 * @param columnScalings Optional column scalings for each of the observation columns
 */
case class KMeansModel private[kmeans] (trainingObservationColumns: Seq[String],
                                        trainingLabelColumn: String,
                                        k: Int,
                                        centroids: Array[Array[Double]],
                                        clusters: Array[String],
                                        clusterSize: Array[Long],
                                        columnScalings: Option[Seq[Double]]) extends Serializable with Model with DaalModel {

  /**
   * Array of observation columns used during training
   */
  def observationColumnsAsArray: Array[String] = trainingObservationColumns.toArray

  /**
   * Array of optional scalings for each of the observation columns
   */
  def columnScalingsAsArray: Array[Double] = columnScalings.getOrElse(List[Double]()).toArray

  /**
   * Predicts the clusters for each data point and distance to every cluster center of the frame using the trained model.
   *
   * @param frame - A frame whose labels are to be predicted. By default, predict is run on the same columns over
   *              which the model is trained.
   * @param observationColumns Column(s) containing the observations whose clusters are to be predicted.
   *                           Default is to predict the clusters over columns the KMeans model was trained on.
   * @param labelColumn Name of output column with index of cluster each observation belongs to.
   * @return A new frame consisting of the existing columns of the frame and the following new columns:
   *         'k' columns : Each of the 'k' columns containing squared distance of that observation to the 'k'th
   *         cluster center.
   *         predicted_cluster column: The cluster assignment for the observation.
   */
  def predict(frame: Frame,
              observationColumns: Option[Seq[String]] = None,
              labelColumn: Option[String] = None): Frame = {
    require(frame != null, "frame is required")

    // Validate DAAL libraries
    DaalUtils.validateDaalLibraries(DaalUtils.daalDynamicLibraries)

    require(labelColumn.isEmpty || StringUtils.isNotEmpty(labelColumn.get), "labelColumn must not be null or empty")
    require(observationColumns.isEmpty || trainingObservationColumns.length == observationColumns.get.length,
      "Number of columns for train and predict should be same")

    val frameRdd = new FrameRdd(frame.schema, frame.rdd)
    val predictObservationColumns = observationColumns.getOrElse(trainingObservationColumns)
    val predictLabelColumn = labelColumn.getOrElse(trainingLabelColumn)
    val resultFrame: FrameRdd = KMeansFunctions.predictKMeansModel(predictObservationColumns, predictLabelColumn, frameRdd, columnScalings, centroids)

    new Frame(resultFrame.rdd, resultFrame.schema)
  }

  /**
   * Predict cluster ID from feature array, used for scoring.
   *
   * @param features feature array
   * @return cluster ID
   */
  private def predictCluster(features: Array[Double]): Int = {
    val context = new DaalContext

    val algorithm = new Batch(context, classOf[lang.Double], Method.lloydDense, k.toLong, 1L)
    val input = new HomogenNumericTable(context, features, features.length, 1L)
    val centroidsTable = DaalTkScoringModelUtils.toDaalNumericTable(centroids)
    centroidsTable.unpack(context)
    algorithm.input.set(InputId.data, input)
    algorithm.input.set(InputId.inputCentroids, centroidsTable)
    algorithm.parameter.setAssignFlag(true)

    val result = algorithm.compute()

    val assignments = result.get(ResultId.assignments).asInstanceOf[HomogenNumericTable]
    val clusterId = DaalTkScoringModelUtils.toDoubleArray(assignments).head
    context.dispose()

    clusterId.toInt
  }

  /**
   * Saves this model to a file
   *
   * @param sc active SparkContext
   * @param path save to path
   */
  override def save(sc: SparkContext, path: String): Unit = {
    val tkMetadata = KMeansModelTkMetaData(trainingObservationColumns,
      trainingLabelColumn,
      k,
      centroids,
      clusters,
      clusterSize,
      columnScalings)
    TkSaveLoad.saveTk(sc, path, KMeansModel.formatId, KMeansModel.currentFormatVersion, tkMetadata)
  }

  /**
   * Scores the given row using the trained DAAL KMeans model
   *
   * @param row Row of input data
   * @return Input row, plus the score
   */
  override def score(row: Array[Any]): Array[Any] = {
    val features: Array[Double] = row.map(y => ScoringModelUtils.asDouble(y))
    val clusterId = predictCluster(features)

    row :+ clusterId
  }

  /**
   * @return DAAL KMeans model metadata
   */
  override def modelMetadata(): ModelMetaData = {
    new ModelMetaData("Intel DAAL KMeans Model", classOf[KMeansModel].getName, classOf[DaalTkModelAdapter].getName, Map())
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
    output :+ Field("score", "Int")
  }
}

/**
 * TK Metadata that will be stored as part of the model
 *
 * @param trainingObservationColumns Column(s) containing the observations whose clusters are to be predicted.
 * @param trainingLabelColumn Name of output column with index of cluster each observation belongs to.
 * @param k Desired number of clusters
 * @param centroids Cluster centroids
 * @param columnScalings Optional column scalings for each of the observation columns
 */
case class KMeansModelTkMetaData(trainingObservationColumns: Seq[String],
                                 trainingLabelColumn: String,
                                 k: Int,
                                 centroids: Array[Array[Double]],
                                 clusters: Array[String],
                                 clusterSize: Array[Long],
                                 columnScalings: Option[Seq[Double]]) extends Serializable