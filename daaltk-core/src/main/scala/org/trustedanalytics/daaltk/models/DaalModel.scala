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

import java.nio.file.{ Files, Path }
import org.apache.commons.io.FileUtils
import org.apache.spark.SparkContext
import org.trustedanalytics.sparktk.models.ScoringModelUtils

trait DaalModel {

  /**
   * Exports a trained DAAL Model to a mar file
   *
   * @param sc active SparkContext
   * @param marSavePath location where the MAR file needs to be saved
   * @return full path to the location of the MAR file
   */
  def exportToMar(sc: SparkContext, marSavePath: String): String = {
    var tmpDir: Path = null
    try {
      tmpDir = Files.createTempDirectory("sparktk-scoring-model")
      save(sc, tmpDir.toString)
      val sourcePath = this.getClass.getProtectionDomain.getCodeSource.getLocation.toString
      ScoringModelUtils.saveToMar(marSavePath, this.getClass.getName, tmpDir, classOf[DaalTkModelAdapter].getName, Some(sourcePath))
    }
    finally {
      sys.addShutdownHook(FileUtils.deleteQuietly(tmpDir.toFile)) // Delete temporary directory on exit
    }
  }

  /**
   * Saves the trained model to a file.  Override from the model class to implement saving.
   */
  def save(sc: SparkContext, path: String): Unit = ???
}
