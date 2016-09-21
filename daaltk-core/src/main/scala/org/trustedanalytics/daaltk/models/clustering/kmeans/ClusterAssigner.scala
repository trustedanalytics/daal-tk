package org.trustedanalytics.daaltk.models.clustering.kmeans

import com.intel.daal.algorithms.kmeans.{ ResultId, InputId, Method, DistributedStep1Local }
import com.intel.daal.data_management.data.HomogenNumericTable
import org.trustedanalytics.sparktk.frame.internal.rdd.FrameRdd
import org.trustedanalytics.daaltk.DaalUtils.withDaalContext
import org.trustedanalytics.sparktk.frame.{ DataTypes, Column, FrameSchema }
import org.trustedanalytics.daaltk.models.tables.{ DistributedNumericTable, IndexedNumericTable }

/**
 * Assign cluster index to each observation in feature table
 *
 * @param featureTable Feature table
 * @param centroids Cluster centroids
 * @param labelColumn Name of output column with index of cluster each observation belongs to
 */
case class ClusterAssigner(featureTable: DistributedNumericTable,
                           centroids: IndexedNumericTable,
                           labelColumn: String) {

  /**
   * Assign cluster index to each observation in feature table
   *
   * @return Frame of cluster assignments
   */
  def assign(): FrameRdd = {
    val schema = FrameSchema(List(Column(labelColumn, DataTypes.int32)))
    var numRows = 0L
    val rdd = featureTable.rdd.map { table =>
      withDaalContext { context =>
        val local = new DistributedStep1Local(context, classOf[java.lang.Double], Method.defaultDense, centroids.numRows)
        local.input.set(InputId.data, table.getUnpackedTable(context))
        local.input.set(InputId.inputCentroids, centroids.getUnpackedTable(context))
        local.parameter.setAssignFlag(true)
        val partialResults = local.compute
        partialResults.pack()

        val result = local.finalizeCompute()
        val assignmentTable = result.get(ResultId.assignments).asInstanceOf[HomogenNumericTable]
        val assignments = IndexedNumericTable(table.index, assignmentTable)
        numRows += assignments.numRows
        assignments
      }.elseError("Could not assign cluster centroids")
    }

    DistributedNumericTable(rdd, numRows).toFrameRdd(schema)
  }

  /**
   * Compute size of predicted clusters
   *
   * @param assignmentFrame Frame with cluster assignments
   * @return Map of cluster names and sizes
   */
  def clusterSizes(assignmentFrame: FrameRdd): Map[String, Long] = {
    //TODO: Use DAAL partial results nObservations to compute cluster sizes
    assignmentFrame.mapRows(row => {
      val clusterId = row.intValue(labelColumn)
      ("Cluster:" + clusterId.toString, 1L)
    }).reduceByKey(_ + _).collect().toMap
  }

}
