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

import java.net.URLClassLoader
import java.util.zip.ZipInputStream

import org.slf4j.LoggerFactory
import org.trustedanalytics.daaltk.models.classification.naive_bayes.NaiveBayesModel
import org.trustedanalytics.daaltk.models.clustering.kmeans.KMeansModel
import org.trustedanalytics.daaltk.models.dimreduction.pca.PcaModel
import org.trustedanalytics.daaltk.models.regression.linear_regression.LinearRegressionModel
import org.trustedanalytics.scoring.interfaces.Model
import org.trustedanalytics.sparktk.models.SparkTkModelAdapter
import org.trustedanalytics.sparktk.saveload.Loaders.LoaderType
import org.trustedanalytics.sparktk.saveload.TkSaveableObject

/**
 * Wrapper that is able to read a MAR file containing daal-tk models and related jars.
 * Loads and returns the Model for scoring.
 */
class DaalTkModelAdapter extends SparkTkModelAdapter {
  private val logger = LoggerFactory.getLogger(this.getClass)

  /**
   * daal-tk model loaders
   */
  private lazy val daalLoaders: Map[String, LoaderType] = {
    val entries: Seq[TkSaveableObject] = List(KMeansModel,
      NaiveBayesModel,
      PcaModel,
      LinearRegressionModel)
    entries.map(e => e.formatId -> e.loadTkSaveableObject _).toMap
  }

  /**
   * Takes a zip input stream with MAR file contents for daal-tk models; then loads and returns a daal-tk model.
   * Passes map of daalLoaders to the spark-tk context during load, to allow spark-tk to load daal-tk models.
   *
   * @param modelZipStreamInput stream with MAR file contents for daal-tk models
   * @return loads and returns the daal-tk model
   */
  override def read(modelZipStreamInput: ZipInputStream, classLoader: URLClassLoader, jsonMap: Map[String, String]): Model = {
    logger.info("daal-tk model adapter read() called")
    Thread.currentThread().setContextClassLoader(classLoader)
    val tc = super.createSimpleContext(modelZipStreamInput)
    tc.load(getModelPath(modelZipStreamInput), Some(daalLoaders)).asInstanceOf[Model]
  }
}
