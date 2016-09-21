package org.trustedanalytics.daaltk

import com.intel.daal.algorithms.{ Result, PartialResult }
import com.intel.daal.services.DaalContext
import org.apache.spark.rdd.RDD

/**
 * Interface for DAAL distributed algorithms
 */
trait DistributedAlgorithm[P, R] {

  /**
   * Compute partial results for algorithm on slave nodes
   *
   * @return RDD of partial results
   */
  def computePartialResults(): RDD[P]

  /**
   * Merge partial results on Spark driver to generate final result
   *
   * @param context DAAL Context
   * @param rdd RDD of partial results
   * @return Final result of algorithm
   */
  def mergePartialResults(context: DaalContext, rdd: RDD[P]): R

}
