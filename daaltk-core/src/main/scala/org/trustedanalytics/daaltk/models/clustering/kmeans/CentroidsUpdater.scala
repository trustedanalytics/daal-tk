package org.trustedanalytics.daaltk.models.clustering.kmeans

import com.intel.daal.algorithms.kmeans._
import com.intel.daal.services.DaalContext
import org.apache.spark.rdd.RDD
import org.trustedanalytics.daaltk.DaalUtils.withDaalContext
import org.trustedanalytics.daaltk.DistributedAlgorithm
import org.trustedanalytics.daaltk.models.tables.{ DistributedNumericTable, IndexedNumericTable }

/**
 * Run one iteration of k-means clustering algorithm to update cluster centroids
 *
 * @param featureTable Feature table
 * @param centroids Input cluster centroids
 * @param labelColumn Name of output column with index of cluster each observation belongs to
 */
case class CentroidsUpdater(featureTable: DistributedNumericTable, centroids: IndexedNumericTable, labelColumn: String)
    extends DistributedAlgorithm[PartialResult, Result] {

  /**
   * Run one iteration of k-means clustering algorithm to update cluster centroids
   *
   * @return Updated centroids
   */
  def updateCentroids(): IndexedNumericTable = {
    withDaalContext { context =>
      val partialResults = computePartialResults()
      val results = mergePartialResults(context, partialResults)
      IndexedNumericTable(0.toLong, results.get(ResultId.centroids))
    }.elseError("Could not update cluster centroids")
  }

  /**
   * Partially update cluster centroids
   *
   * @return RDD of partial k-means and optional cluster assignments
   */
  override def computePartialResults(): RDD[PartialResult] = {
    featureTable.rdd.map { table =>
      withDaalContext { context =>
        val local = new DistributedStep1Local(context, classOf[java.lang.Double], Method.defaultDense, centroids.numRows)
        local.input.set(InputId.data, table.getUnpackedTable(context))
        local.input.set(InputId.inputCentroids, centroids.getUnpackedTable(context))
        local.parameter.setAssignFlag(false)

        val partialResult = local.compute
        partialResult.pack()
        partialResult
      }.elseError("Could not compute partial results for updating cluster centroids")
    }
  }

  /**
   * Merge partial results of K-means clustering to compute cluster centroids
   *
   * @param context DAAL context
   * @param partsRdd RDD of partial results
   * @return Updated cluster centroids
   */
  override def mergePartialResults(context: DaalContext, partsRdd: RDD[PartialResult]): Result = {
    val partialResults = partsRdd.collect()
    val master = new DistributedStep2Master(context, classOf[java.lang.Double], Method.defaultDense, centroids.numRows)

    for (value <- partialResults) {
      value.unpack(context)
      master.input.add(DistributedStep2MasterInputId.partialResults, value)
    }
    master.compute

    val result = master.finalizeCompute
    result
  }

}

