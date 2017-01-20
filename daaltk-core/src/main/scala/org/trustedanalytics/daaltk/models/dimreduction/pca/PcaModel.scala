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
package org.trustedanalytics.daaltk.models.dimreduction.pca

import breeze.linalg.DenseVector
import org.apache.commons.lang.StringUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.Row.merge
import org.json4s.JsonAST.JValue
import org.trustedanalytics.daaltk.models.{ DaalModel, DaalTkModelAdapter }
import org.trustedanalytics.sparktk.TkContext
import org.trustedanalytics.sparktk.frame._
import org.trustedanalytics.sparktk.frame.internal.rdd.FrameRdd
import org.trustedanalytics.sparktk.models.MatrixImplicits._
import org.trustedanalytics.sparktk.models.dimreduction.pca.PrincipalComponentsFunctions
import org.trustedanalytics.sparktk.saveload.{ SaveLoad, TkSaveLoad, TkSaveableObject }
import org.trustedanalytics.scoring.interfaces.{ ModelMetaData, Field, Model }
import org.trustedanalytics.sparktk.models.ScoringModelUtils
import org.apache.spark.mllib.linalg.{ DenseVector => MllibDenseVector, DenseMatrix => MllibDenseMatrix }

object PcaModel extends TkSaveableObject {
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
  override def formatId: String = "DaalPrincipalComponentsModel"

  /**
   * Build Intel DAAL principal components model using the observation columns.
   *
   * @param frame The frame containing the data to train on.
   * @param observationColumns List of columns(s) containing the observations.
   * @param meanCentered Option to mean center the columns.
   * @param k Principal component count. Default is the number of observation columns.
   */
  def train(frame: Frame,
            observationColumns: Seq[String],
            meanCentered: Boolean = true,
            k: Option[Int] = None): PcaModel = {
    require(frame != null, "frame is required")
    require(observationColumns != null && observationColumns.length > 0, "observations columns must not be null nor empty.")
    require(observationColumns.forall(StringUtils.isNotEmpty(_)), "observation columns names cannot be null or empty")
    require(k.isEmpty || k.get <= observationColumns.length, "k must be less than or equal to number of observation columns")

    frame.schema.requireColumnsAreVectorizable(observationColumns)

    val frameRdd = new FrameRdd(frame.schema, frame.rdd)
    val trainK = k.getOrElse(observationColumns.length)

    PcaModel(SvdAlgorithm(frameRdd, observationColumns, meanCentered).compute(trainK, computeU = false))
  }

  /**
   * Create a DAAL PrincipalComponentsModel, given the specified metadata.
   */
  def loadTkSaveableObject(sc: SparkContext, path: String, formatVersion: Int, tkMetadata: JValue): Any = {
    validateFormatVersion(formatVersion, validFormatVersions: _*)
    val m: PrincipalComponentsTkMetaData = SaveLoad.extractFromJValue[PrincipalComponentsTkMetaData](tkMetadata)

    // Create SvdData
    val svdData = SvdData(m.k,
      m.observationColumns,
      m.meanCentered,
      new MllibDenseVector(m.meanVector),
      new MllibDenseVector(m.singularValues),
      new DenseMatrix(m.vFactorRows, m.vFactorCols, m.rightSingularVectors),
      m.leftSingularMatrix)

    // Create PrincipalComponentsModel to return
    PcaModel(svdData)
  }

  /**
   * Load a DAAL prinicpal components model from the given path
   *
   * @param tc TkContext
   * @param path location
   * @return
   */
  def load(tc: TkContext, path: String): PcaModel = {
    tc.load(path).asInstanceOf[PcaModel]
  }
}

/**
 * A PCA Model, holding all the important parameters
 *
 * @param svdData Model data for Intel DAAL Singular Value Decomposition (SVD)
 */
case class PcaModel(svdData: SvdData) extends Serializable with Model with DaalModel {
  /**
   * Observation columns from the training data
   */
  def observationColumns: Array[String] = svdData.observationColumns.toArray

  /**
   * Indicator whether the columns were mean centered for training.
   */
  def meanCentered: Boolean = svdData.meanCentered

  /**
   * Principal component count
   */
  def k: Int = svdData.k

  /**
   * Means of the columns
   */
  def columnMeans: Array[Double] = svdData.meanVector.toArray

  /**
   * Singular values of the specified columns in the input frame
   */
  def singularValues: Array[Double] = svdData.singularValues.toArray

  /**
   * Right singular vectors of the specified columns in the input frame
   */
  def rightSingularVectors: Array[Array[Double]] = {
    val lists = svdData.rightSingularVectors.toListOfList()

    lists.map(list => list.toArray).toArray
  }

  /**
   * Predicting on a dataframe's columns using a PrincipalComponents Model.
   *
   * @param frame frame on which to base predictions and to which the predictions will be added.
   * @param meanCentered Option to mean center the columns. Default is true.
   * @param tSquaredIndex Indicator for whether the t-square index is to be computed. Default is false
   * @param observationColumns The list of observation column name(s) to be used for prediction.  Default is the list
   *                           of column name(s) used to train the model.
   * @param c The number of principal components to be predicted.  'c' cannot be greater than the count used to train
   *          the model.  Default is the count used to train the model.
   * @return A frame with existing columns and the following additional columns:
   *         'c' additional columns: containing the projections of V on the frame 't_squared_index': column storing
   *         the t-squared-index value, if requested.
   */
  def predict(frame: Frame,
              meanCentered: Boolean = true,
              tSquaredIndex: Boolean = false,
              observationColumns: Option[Seq[String]] = None,
              c: Option[Int] = None): Frame = {

    // Validate arguments
    require(frame != null, "frame is required")
    if (svdData.meanCentered)
      require(meanCentered, "Cannot mean center the predict frame if the train frame was not mean centered.")
    if (observationColumns.isDefined)
      require(svdData.observationColumns.length == observationColumns.get.length, "Number of columns for train and predict should be same")
    if (c.isDefined)
      require(svdData.k >= c.get, "Number of components must be at most the number of components trained on")
    if (meanCentered)
      require(this.meanCentered, "Cannot mean center the predict frame if the train frame was not mean centered.")
    val predictC = c.getOrElse(svdData.k)
    val predictColumns = observationColumns.getOrElse(svdData.observationColumns)
    val frameRdd = new FrameRdd(frame.schema, frame.rdd).zipWithIndex().map { case (row, index) => (index, row) }
    require(predictColumns.length == svdData.observationColumns.length, "Number of columns for train and predict should be same")
    require(predictC <= this.k, s"Number of components ($predictC) must be at most the number of components trained on ($this.k)")
    predictColumns.map(columnName => frame.schema.requireColumnIsNumerical(columnName))

    //TODO: Update predict method once DAAL supports linear algebra operations on its numeric tables
    // Predict principal components and optional T-squared index
    val indexedRowMatrix = PrincipalComponentsFunctions.toIndexedRowMatrix(
      frameRdd,
      frame.schema,
      predictColumns,
      meanCentered,
      columnMeans)
    val principalComponents = PrincipalComponentsFunctions.computePrincipalComponents(svdData.rightSingularVectors, predictC, indexedRowMatrix)

    val pcaColumns = for (i <- 1 to predictC) yield Column("p_" + i.toString, DataTypes.float64)
    val (componentColumns, components) = tSquaredIndex match {
      case true => {
        val tSquareMatrix = PrincipalComponentsFunctions.computeTSquaredIndex(principalComponents, svdData.singularValues, predictC)
        val tSquareColumn = Column("t_squared_index", DataTypes.float64)
        (pcaColumns :+ tSquareColumn, tSquareMatrix)
      }
      case false => (pcaColumns, principalComponents)
    }

    val componentRows = components.rows.map(row => (row.index, Row.fromSeq(row.vector.toArray.toSeq)))

    val joinedFrame = frameRdd.join(componentRows).map({ case (index, (rowA, rowB)) => (merge(rowA, rowB)) })
    val componentFrame = new FrameRdd(FrameSchema(frame.schema.columns ++ componentColumns), joinedFrame)
    new Frame(componentFrame, componentFrame.schema)
  }

  /**
   * Saves this model to a file
   *
   * @param sc active SparkContext
   * @param path save to path
   */
  override def save(sc: SparkContext, path: String): Unit = {
    val tkMetadata = PrincipalComponentsTkMetaData(svdData.k,
      svdData.observationColumns,
      svdData.meanCentered,
      svdData.meanVector.toArray,
      svdData.singularValues.toArray,
      svdData.rightSingularVectors.toArray,
      svdData.rightSingularVectors.numRows,
      svdData.rightSingularVectors.numCols,
      svdData.leftSingularMatrix)
    TkSaveLoad.saveTk(sc, path, PcaModel.formatId, PcaModel.currentFormatVersion, tkMetadata)
  }

  /**
   * Scores the given row using the trained DAAL Principal Components model
   *
   * @param row Row of input data
   * @return Input row, plus the score
   */
  override def score(row: Array[Any]): Array[Any] = {
    val x: Array[Double] = row.map(value => ScoringModelUtils.asDouble(value))
    var inputVector = new MllibDenseVector(x)
    if (meanCentered) {
      val meanCenteredVector: Array[Double] = (new DenseVector(x) - new DenseVector(columnMeans.toArray)).toArray
      inputVector = new MllibDenseVector(meanCenteredVector)
    }
    val y = new MllibDenseMatrix(1, inputVector.size, inputVector.toArray).multiply(svdData.rightSingularVectors.asInstanceOf[MllibDenseMatrix])
    val yArray: Array[Double] = y.values
    var t_squared_index: Double = 0.0
    for (i <- 0 until k) {
      if (singularValues(i) > 0)
        t_squared_index += ((yArray(i) * yArray(i)) / (singularValues(i) * singularValues(i)))
    }
    row ++ Array(y.values.toList, t_squared_index)
  }

  /**
   * @return DAAL Principal Components model metadata
   */
  override def modelMetadata(): ModelMetaData = {
    new ModelMetaData("Intel DAAL Principal Components Model",
      classOf[PcaModel].getName,
      classOf[DaalTkModelAdapter].getName,
      Map())
  }

  /**
   * @return fields containing the input names and their data types
   */
  override def input(): Array[Field] = {
    svdData.observationColumns.map(name => Field(name, "Double")).toArray
  }

  /**
   * @return fields containing the input names and their data types along with the output and its data type
   */
  override def output(): Array[Field] = {
    var output = input()
    output = output :+ Field("principal_components", "List[Double]")
    output :+ Field("t_squared_index", "Double")
  }
}

/**
 * TK Metadata that will be stored as part of the model
 *
 * @param k Principal component count
 * @param observationColumns Handle to the observation columns of the data frame
 * @param meanCentered Indicator whether the columns were mean centered for training
 * @param meanVector Means of the columns
 * @param singularValues Singular values of the specified columns in the input frame
 * @param rightSingularVectors Right singular vectors of the specified columns in the input frame
 * @param vFactorRows Number of rows in vFactor matrix
 * @param vFactorCols Number of columns in vFactor matrix
 * @param leftSingularMatrix Optional RDD with left singular vectors of the specified columns in the input frame
 */
case class PrincipalComponentsTkMetaData(k: Int,
                                         observationColumns: Seq[String],
                                         meanCentered: Boolean,
                                         meanVector: Array[Double],
                                         singularValues: Array[Double],
                                         rightSingularVectors: Array[Double],
                                         vFactorRows: Int,
                                         vFactorCols: Int,
                                         leftSingularMatrix: Option[RDD[Vector]]) extends Serializable