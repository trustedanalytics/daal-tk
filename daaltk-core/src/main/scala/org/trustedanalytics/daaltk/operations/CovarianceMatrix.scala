package org.trustedanalytics.daaltk.operations

import org.trustedanalytics.sparktk.frame.internal.rdd.FrameRdd
import org.trustedanalytics.sparktk.frame.{ SchemaHelper, Frame, DataTypes }
import org.trustedanalytics.sparktk.frame.DataTypes._
import com.intel.daal.algorithms.covariance.ResultId

object CovarianceMatrix {

  /**
   * Uses Intel Data Analytics and Acceleration Library (DAAL) to compute covariance matrix.
   *
   * Notes
   * -----
   * This function applies only to columns containing numerical data.
   *
   * @param frame Frame of data
   * @param dataColumnNames The names of the column from which to compute the matrix. Names should refer to a single
   *                        column of type vector, or two or more columns of numeric scalars.  If no column names are
   *                        provided, all columns in the frame will be used.
   * @return Frame with a matrix with the covariance values for the columns.
   */
  def covarianceMatrix(frame: Frame, dataColumnNames: Option[Seq[String]]): Frame = {
    require(frame != null, "Frame is required")
    val columnNames = dataColumnNames.getOrElse(frame.schema.columnNames)
    frame.schema.requireColumnsAreVectorizable(columnNames)

    // compute covariance
    val outputColumnDataType = frame.schema.columnDataType(columnNames.head)
    val outputVectorLength: Option[Long] = outputColumnDataType match {
      case vector(length) => Some(length)
      case _ => None
    }

    val frameRdd = new FrameRdd(frame.schema, frame.rdd)
    val covarianceRdd = DaalCovarianceFunctions.covarianceMatrix(frameRdd, columnNames.toList,
      ResultId.covariance, outputVectorLength)
    val outputSchema = SchemaHelper.create(columnNames, DataTypes.float64, outputVectorLength)

    new Frame(covarianceRdd, outputSchema)
  }
}