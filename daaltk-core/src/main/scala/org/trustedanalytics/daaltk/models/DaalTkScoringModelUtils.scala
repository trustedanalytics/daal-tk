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
package org.trustedanalytics.daaltk.models

import java.nio.DoubleBuffer
import com.intel.daal.data_management.data.{ HomogenNumericTable, NumericTable }
import com.intel.daal.services.DaalContext

object DaalTkScoringModelUtils {

  /**
   * Convert matrix to Intel DAAL homogeneous numeric table
   */
  def toDaalNumericTable(matrix: Array[Array[Double]]): HomogenNumericTable = {
    require(matrix != null && matrix.length > 0, "Array must not be null or empty")
    val context = new DaalContext()
    val numRows = matrix.length
    val array = matrix.flatten
    val table = new HomogenNumericTable(context, array, array.length / numRows, numRows)
    table.pack()
    context.dispose()
    table
  }

  /**
   * Convert DAAL numeric table into an array of array of doubles
   *
   * Each feature vector (row) in the numeric table is converted into an array of doubles
   *
   * @return Array of array of doubles
   */
  def toArrayOfDoubleArray(table: NumericTable): Array[Array[Double]] = {
    val context = new DaalContext()
    table.unpack(context)
    val numRows = table.getNumberOfRows.toInt
    val numCols = table.getNumberOfColumns.toInt
    val buffer = DoubleBuffer.allocate(numRows * numCols)
    val doubleBuffer = table.getBlockOfRows(0, numRows, buffer)

    val arrays = new Array[Array[Double]](numRows)
    for (i <- 0 until numRows) {
      val rowArray = new Array[Double](numCols)
      for (j <- 0 until numCols) {
        rowArray(j) = doubleBuffer.get(i * numCols + j)
      }
      arrays(i) = rowArray
    }
    context.dispose()

    arrays
  }

  /**
   * Convert DAAL numeric table into an array of doubles
   *
   * Each feature vector (row) in the numeric table is converted into an array of doubles
   *
   * @return Array of array of doubles
   */
  def toDoubleArray(table: NumericTable): Array[Double] = {
    toArrayOfDoubleArray(table).flatten
  }
}
