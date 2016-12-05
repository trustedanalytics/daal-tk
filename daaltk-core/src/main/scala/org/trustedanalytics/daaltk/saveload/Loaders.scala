package org.trustedanalytics.daaltk.saveload

import org.trustedanalytics.daaltk.models.classification.naive_bayes.NaiveBayesModel
import org.trustedanalytics.daaltk.models.clustering.kmeans.KMeansModel
import org.trustedanalytics.daaltk.models.dimreduction.pca.PcaModel
import org.trustedanalytics.daaltk.models.regression.linear_regression.LinearRegressionModel
import org.trustedanalytics.sparktk.saveload.TkSaveableObject
import org.trustedanalytics.sparktk.saveload.Loaders.LoaderType

object Loaders {

  /**
   * Function is used to get daal-tk's Loaders.  When spark-tk is loading something, and the item's format does not
   * match a known spark-tk Loader, then it calls out to it's other libraries (including daal-tk) to get their
   * loaders, to try to find a match.
   *
   * @return Map of DAAL models that can be loaded
   */
  def getLoaders(): Map[String, LoaderType] = {
    loaders
  }

  /**
   * Registry of all the DAAL loaders
   *
   * If you have an class that wants to play TkSaveLoad, it needs an entry in here:
   *
   * formatId -> loader function
   */
  private lazy val loaders: Map[String, LoaderType] = {
    val entries: Seq[TkSaveableObject] = List(NaiveBayesModel,
      KMeansModel,
      PcaModel,
      LinearRegressionModel)
    entries.map(e => e.formatId -> e.loadTkSaveableObject _).toMap
  }

}