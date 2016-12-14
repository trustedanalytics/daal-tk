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
package org.trustedanalytics.daaltk.models.tables

import com.intel.daal.data_management.data.HomogenNumericTable
import org.trustedanalytics.sparktk.frame.internal.rdd.FrameRdd
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.trustedanalytics.daaltk.DaalUtils.withDaalContext
import scala.collection.mutable.ArrayBuffer

/**
 * Distributed DAAL numeric table with features and labels
 *
 * @param tableRdd RDD of indexed numeric table
 * @param numRows Number of rows in table
 */
case class DistributedLabeledTable(tableRdd: RDD[IndexedLabeledTable],
                                   numRows: Long) extends DistributedTable(tableRdd, numRows) {
  require(tableRdd != null, "DAAL labeled table RDD must not be null")
  val numFeatureCols = tableRdd.first().features.numCols
  val numLabelCols = tableRdd.first().labels.numCols
}

/**
 * Factory object for creating labeled tables
 */
object DistributedLabeledTable {

  /**
   * Create distributed numeric table with features and labels from Vector RDD
   *
   * @param vectorRdd Vector RDD
   * @param splitIndex Column index at which to split vector into features and labels
   *
   * @return Distributed labeled table
   */
  def createTable(vectorRdd: RDD[Vector], splitIndex: Int): DistributedLabeledTable = {
    val first: Vector = vectorRdd.first()

    val numCols = first.size
    val numFeatureCols = splitIndex
    val numLabelCols = numCols - splitIndex

    val tableRdd = vectorRdd.mapPartitionsWithIndex {
      case (i, iter) =>
        val indexedTable: IndexedLabeledTable = withDaalContext { context =>
          val featureBuf = new ArrayBuffer[Double]()
          val labelBuf = new ArrayBuffer[Double]()
          var numRows = 0L

          while (iter.hasNext) {
            val array = iter.next().toArray
            featureBuf ++= array.slice(0, splitIndex)
            labelBuf ++= array.slice(splitIndex, numCols)
            numRows += 1
          }

          val featureTable = new IndexedNumericTable(i, new HomogenNumericTable(context,
            featureBuf.toArray, numFeatureCols, numRows))
          val labelTable = new IndexedNumericTable(i, new HomogenNumericTable(context,
            labelBuf.toArray, numLabelCols, numRows))
          IndexedLabeledTable(featureTable, labelTable)
        }.elseError("Could not create numeric table from Vector RDD")

        Array(indexedTable).toIterator
    }.filter(_.features.numRows > 0)

    val totalRows = tableRdd.map(table => table.features.numRows.toLong).sum().toLong
    DistributedLabeledTable(tableRdd, totalRows)
  }

  /**
   * Create distributed numeric table with features and labels from frame
   *
   * @param frameRdd Input frame
   * @param featureColumns List of feature columns
   * @param labelColumns List of label columns
   *
   * @return Distributed labeled table
   */
  def createTable(frameRdd: FrameRdd, featureColumns: Seq[String], labelColumns: Seq[String]): DistributedLabeledTable = {
    createTable(frameRdd.toDenseVectorRdd(featureColumns ++ labelColumns), featureColumns.size)
  }
}
