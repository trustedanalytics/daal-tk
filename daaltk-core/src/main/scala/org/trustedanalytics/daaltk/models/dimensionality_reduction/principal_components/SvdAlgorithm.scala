package org.trustedanalytics.daaltk.models.dimensionality_reduction.principal_components

import java.util.Arrays
import com.intel.daal.algorithms.svd._
import com.intel.daal.data_management.data.{ DataCollection, HomogenNumericTable }
import com.intel.daal.services.DaalContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.trustedanalytics.daaltk.DaalUtils._
import org.trustedanalytics.daaltk.DistributedAlgorithm
import org.trustedanalytics.daaltk.models.tables.DaalConversionImplicits._
import org.trustedanalytics.daaltk.models.tables.{ DistributedNumericTable, IndexedNumericTable }
import org.trustedanalytics.sparktk.frame.internal.rdd.FrameRdd

/**
 * Partial results of singular value decomposition algorithm
 *
 * @param tableIndex Unique table index
 * @param vMatrixDataCollection Partial results for computing right singular matrix
 * @param uMatrixDataCollection Partial results for computing left singular matrix
 */
case class SvdPartialResults(tableIndex: Long, vMatrixDataCollection: DataCollection, uMatrixDataCollection: DataCollection)

/**
 * Results of singular value decomposition algorithm aggregated at the master
 *
 * @param result Result object with right singular values and right singular matrix
 * @param partialResult Partial results generated at master for computing left singular matrix
 */
case class SvdMasterResult(result: Result, partialResult: DistributedStep2MasterPartialResult)

/**
 * Computes the singular value decomposition of input frame using Intel DAAL
 *
 * @param frameRdd Input frame
 * @param observationColumns List of columns(s) containing the observations.
 * @param meanCentered Option to mean center the columns.
 */
case class SvdAlgorithm(frameRdd: FrameRdd,
                        observationColumns: Seq[String],
                        meanCentered: Boolean) extends DistributedAlgorithm[SvdPartialResults, SvdMasterResult] {

  private val vectorRdd = if (meanCentered) frameRdd.toMeanCenteredDenseVectorRdd(observationColumns)
  else frameRdd.toDenseVectorRdd(observationColumns)

  private val distributedTable = DistributedNumericTable.createTable(vectorRdd)

  /**
   * Computes the singular value decomposition of  input frame
   *
   * @param k Principal component count
   * @param computeU If true, compute left singular matrix
   * @return Model data with right singular values, right singular matrix,
   *         and optional left singular matrix
   */
  def compute(k: Int, computeU: Boolean = false): SvdData = {
    require(k > 0 && k <= observationColumns.length, "k must be smaller than the number of observation columns")

    val modelData = withDaalContext { context =>
      val partialResults = computePartialResults()
      val svdMasterResult = mergePartialResults(context, partialResults)
      val columnStatistics = frameRdd.columnStatistics(observationColumns)
      val singularValues = getSingularValues(svdMasterResult, k)
      val rightSingularMatrix = getRightSingularMatrix(svdMasterResult, k)
      val leftSingularMatrix = computeLeftSingularMatrix(svdMasterResult.partialResult,
        partialResults, computeU)

      SvdData(k, observationColumns, meanCentered, columnStatistics.mean,
        singularValues, rightSingularMatrix, leftSingularMatrix)
    }.elseError("Could not compute singular value decomposition")
    modelData
  }

  /**
   * Compute partial results for singular values, right and left singular matrices
   *
   * @return RDD of partial SVD results
   */
  override def computePartialResults(): RDD[SvdPartialResults] = {
    distributedTable.rdd.map(table => {
      withDaalContext { context =>
        val svdLocal = new DistributedStep1Local(context, classOf[java.lang.Double], Method.defaultDense)
        svdLocal.input.set(InputId.data, table.getUnpackedTable(context))
        val partialResult = svdLocal.compute
        val vMatrixDataCollection = partialResult.get(PartialResultId.outputOfStep1ForStep2)
        val uMatrixDataCollection = partialResult.get(PartialResultId.outputOfStep1ForStep3)
        vMatrixDataCollection.pack()
        uMatrixDataCollection.pack()
        SvdPartialResults(table.index, vMatrixDataCollection, uMatrixDataCollection)
      }.elseError("Could not compute right singular matrix")
    })
  }

  /**
   * Merge partial PCA results and compute singular values and right singular matrix
   *
   * @param context DAAL context
   * @param rdd RDD of partial results
   * @return Result object with singular values and right singular matrix
   */
  override def mergePartialResults(context: DaalContext,
                                   rdd: RDD[SvdPartialResults]): SvdMasterResult = {
    val partialResults = rdd.map(p => (p.tableIndex, p.vMatrixDataCollection)).collect()
    val svdMaster = new DistributedStep2Master(context, classOf[java.lang.Double], Method.defaultDense)

    partialResults.foreach {
      case ((i, dataCollection)) =>
        dataCollection.unpack(context)
        svdMaster.input.add(DistributedStep2MasterInputId.inputOfStep2FromStep1, i.toInt, dataCollection)
    }

    val svdMasterPartialResult = svdMaster.compute
    val svdMasterResult = svdMaster.finalizeCompute
    SvdMasterResult(svdMasterResult, svdMasterPartialResult)
  }

  /**
   * Compute left singular matrix
   *
   * @param svdMasterPartialResult Partial results from master
   * @param rdd Partial results with data collection for computing left singular matrix
   * @return RDD with left singular matrix
   */
  private def computeLeftSingularMatrix(svdMasterPartialResult: DistributedStep2MasterPartialResult,
                                        rdd: RDD[SvdPartialResults],
                                        computeU: Boolean = false): Option[RDD[Vector]] = {

    if (!computeU) return None

    val partialResultBcast = rdd.sparkContext.broadcast(svdMasterPartialResult)

    val leftSingularMatrixRdd = rdd.flatMap(svdPartialResults => {
      val vectorIterator = withDaalContext { context =>
        val masterResult = partialResultBcast.value
        masterResult.unpack(context)

        val uMatrixPartialResult = masterResult.get(DistributedPartialResultCollectionId.outputOfStep2ForStep3)
        val svdMasterDataCollection = uMatrixPartialResult.get(svdPartialResults.tableIndex.toInt).asInstanceOf[DataCollection]
        val uMatrixDataCollection = svdPartialResults.uMatrixDataCollection
        uMatrixDataCollection.unpack(context)

        val svdLocal = new DistributedStep3Local(context, classOf[java.lang.Double], Method.defaultDense)
        svdLocal.input.set(DistributedStep3LocalInputId.inputOfStep3FromStep1, uMatrixDataCollection)
        svdLocal.input.set(DistributedStep3LocalInputId.inputOfStep3FromStep2, svdMasterDataCollection)
        svdLocal.compute()
        val result = svdLocal.finalizeCompute()

        val uMatrix = result.get(ResultId.leftSingularMatrix).asInstanceOf[HomogenNumericTable]
        val leftSingularMatrix = IndexedNumericTable(svdPartialResults.tableIndex, uMatrix)
        leftSingularMatrix.toVectorIterator(context)
      }.elseError("Could not compute left singular matrix")
      vectorIterator
    })

    Some(leftSingularMatrixRdd)
  }

  /**
   * Get right singular matrix from SVD master result
   * @param results SVD master result
   * @param k Principal component count
   * @return Right singular matrix
   */
  private def getRightSingularMatrix(results: SvdMasterResult, k: Int): Matrix = {
    val rightSingularMatrix = results.result.get(
      ResultId.rightSingularMatrix).toMatrix(k)
    rightSingularMatrix
  }

  /**
   * Get right singular values from SVD master result
   * @param results SVD master result
   * @param k Principal component count
   * @return Right singular values
   */
  private def getSingularValues(results: SvdMasterResult, k: Int): Vector = {
    val singularValues = results.result.get(ResultId.singularValues).toDoubleArray()
    Vectors.dense(Arrays.copyOfRange(singularValues, 0, k))
  }
}

/**
 * Command for loading model data into existing model in the model database.
 * @param k Principal component count
 * @param observationColumns Handle to the observation columns of the data frame
 * @param meanCentered Indicator whether the columns were mean centered for training
 * @param meanVector Means of the columns
 * @param singularValues Singular values of the specified columns in the input frame
 * @param vFactor Right singular vectors of the specified columns in the input frame
 */
case class PrincipalComponentsData(k: Int,
                                   observationColumns: Seq[String],
                                   meanCentered: Boolean,
                                   meanVector: Vector,
                                   singularValues: Vector,
                                   vFactor: Matrix) {
  require(observationColumns != null && observationColumns.nonEmpty, "observationColumns must not be null nor empty")
  require(k >= 1, "number of Eigen values to use must be greater than equal to 1")
}

/**
 * Model data for Intel DAAL Singular Value Decomposition (SVD)
 *
 * @param k Principal component count
 * @param observationColumns Handle to the observation columns of the data frame
 * @param meanCentered Indicator whether the columns were mean centered for training
 * @param meanVector Means of the columns
 * @param singularValues Singular values of the specified columns in the input frame
 * @param vFactor Right singular vectors of the specified columns in the input frame
 * @param leftSingularMatrix Optional RDD with left singular vectors of the specified columns in the input frame
 */
case class SvdData(k: Int,
                   observationColumns: Seq[String],
                   meanCentered: Boolean,
                   meanVector: Vector,
                   singularValues: Vector,
                   vFactor: Matrix,
                   leftSingularMatrix: Option[RDD[Vector]]) {
  require(observationColumns != null && observationColumns.nonEmpty, "observationColumns must not be null nor empty")
  require(k >= 1, "number of Eigen values to use must be greater than equal to 1")
  require(k <= observationColumns.length,
    "k must be less than or equal to number of observation columns")

  /**
   * Convert Intel DAAL SVD data to Principal Components model data
   */
  def toPrincipalComponentsData: PrincipalComponentsData = {
    PrincipalComponentsData(k,
      observationColumns,
      meanCentered,
      meanVector,
      singularValues,
      vFactor)
  }
}

