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

import com.intel.daal.algorithms.kmeans._
import com.intel.daal.algorithms.kmeans.init.InitMethod
import com.intel.daal.data_management.data.HomogenNumericTable
import org.trustedanalytics.sparktk.frame.internal.rdd.FrameRdd
import org.trustedanalytics.sparktk.frame.internal.rdd.VectorUtils._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.trustedanalytics.sparktk.frame.{ FrameSchema, Column, DataTypes }
import org.trustedanalytics.daaltk.models.tables.{ DistributedNumericTable, IndexedNumericTable }
import org.apache.mahout.math.{ DenseVector => MahoutDenseVector }
import org.trustedanalytics.daaltk.models.tables.DaalConversionImplicits._
import scala.collection.mutable.ListBuffer

object KMeansFunctions extends Serializable {

  /**
   * Train K-means clustering model
   *
   * @param frameRdd Input frame
   * @param observationColumns The columns containing the observations
   * @param columnScalings Optional column scalings for each of the observation columns. The scaling value is
   *                       multiplied by the corresponding value in the observation column.
   * @param k Desired number of clusters.
   * @param maxIterations Number of iterations for which the algorithm should run.
   * @param labelColumn Optional name of output column with index of cluster each observation belongs to.
   * @return Trained k-means model
   */
  def trainKMeansModel(frameRdd: FrameRdd,
                       observationColumns: Seq[String],
                       columnScalings: Option[List[Double]],
                       k: Int,
                       maxIterations: Int,
                       labelColumn: String): KMeansTrainReturn = {

    val vectorRdd = createVectorRdd(frameRdd, observationColumns, columnScalings)
    val table = DistributedNumericTable.createTable(vectorRdd)
    table.cache()

    // Iteratively update cluster centroids
    val args = CentroidsInitArgs(k)
    var centroids = CentroidsInitializer(table, args).initializeCentroids()
    for (i <- 1 to maxIterations) {
      centroids = CentroidsUpdater(table, centroids, labelColumn).updateCentroids()
    }

    // Create frame with cluster assignments
    val clusterAssigner = ClusterAssigner(table, centroids, labelColumn)
    val assignmentFrame = clusterAssigner.assign()
    val clusterSizes = clusterAssigner.clusterSizes(assignmentFrame)

    //Get dictionary with centroids
    val centroidsMap = centroids.table.toArrayOfDoubleArray().zipWithIndex.map {
      case (centroid, i) =>
        ("Cluster:" + i.toString, centroid)
    }.toMap
    table.unpersist()

    KMeansTrainReturn(centroidsMap, clusterSizes)
  }

  /**
   * Predict cluster assignments for KMeans model
   *
   * @param observationColumns Column(s) containing the observations whose clusters are to be predicted.
   * @param labelColumn Name of output column with index of cluster each observation belongs to.
   * @param frameRdd Input frame
   * @param columnScalings Optional column scalings for each of the observation columns
   * @param centroids Cluster centroids
   * @return Frame with cluster assignments
   */
  def predictKMeansModel(observationColumns: Seq[String],
                         labelColumn: String,
                         frameRdd: FrameRdd,
                         columnScalings: Option[Seq[Double]],
                         centroids: Array[Array[Double]]): FrameRdd = {

    // Compute cluster assignments
    val vectorRdd = createVectorRdd(frameRdd, observationColumns, columnScalings)
    vectorRdd.cache()
    val table = DistributedNumericTable.createTable(vectorRdd)
    val centroidsTable = IndexedNumericTable.createTable(0L, centroids)

    // Create assignment and cluster distances frame
    val assignFrame = ClusterAssigner(table, centroidsTable, labelColumn).assign()
    val distanceFrame = computeClusterDistances(vectorRdd, centroids)
    frameRdd.zipFrameRdd(distanceFrame).zipFrameRdd(assignFrame)
  }

  /**
   * Compute distances to cluster centroids for each observation in Vector RDD
   *
   * @param vectorRdd Vector RDD
   * @param centroids Cluster centroids
   * @return Frame with 'k' columns with squared distance of each observation to 'k'th cluster center
   */
  def computeClusterDistances(vectorRdd: RDD[Vector], centroids: Array[Array[Double]]): FrameRdd = {
    val rowRdd: RDD[Row] = vectorRdd.map(vector => {
      val distances: Array[Any] = centroids.map(centroid => {
        toMahoutVector(vector).getDistanceSquared(new MahoutDenseVector(centroid))
      })
      new GenericRow(distances)
    })

    val columns = new ListBuffer[Column]()
    for (i <- 0 until centroids.length) {
      val colName = "distance_from_cluster_" + i.toString
      columns += Column(colName, DataTypes.float64)
    }
    new FrameRdd(FrameSchema(columns.toList), rowRdd)
  }

  /**
   * Get goal function for k-means
   *
   * The goal function is the sum of distances of observations to their closest cluster center
   *
   * @param result k-means result
   * @return sum of distances of observations to their closest cluster center
   */
  private def getGoalFunction(result: Result): Double = {
    //TODO: Goal function is returning zero in DAAL 2016.0.109. Revisit after upgrade
    val goal = result.get(ResultId.goalFunction).asInstanceOf[HomogenNumericTable]
    val arr = goal.getDoubleArray
    if (arr.size > 0) {
      arr(0)
    }
    else {
      throw new RuntimeException("Unable to calculate goal function for k-means clustering")
    }
  }

  /**
   * Create Vector RDD from observation columns in frame
   *
   * @param frameRdd Input frame
   * @param observationColumns Observation columns
   * @param columnScalings Optional column scalings for each of the observation columns
   * @return Vector RDD
   */
  private def createVectorRdd(frameRdd: FrameRdd,
                              observationColumns: Seq[String],
                              columnScalings: Option[Seq[Double]] = None): RDD[Vector] = {
    val vectorRdd = columnScalings match {
      case Some(scalings) => frameRdd.toDenseVectorRddWithWeights(observationColumns, scalings)
      case _ => frameRdd.toDenseVectorRdd(observationColumns)
    }
    vectorRdd
  }

}

/**
 * Return object when training a DAAL KMeansModel
 * @param centroids A dictionary containing the cluster Id and centroids
 * @param clusterSize A dictionary of cluster names and sizes
 */
case class KMeansTrainReturn(centroids: Map[String, Array[Double]], clusterSize: Map[String, Long])
