package org.trustedanalytics.daaltk.testutils

import org.apache.spark.SparkContext
import org.scalatest.{ BeforeAndAfterAll, WordSpec }

trait TestingSparkContextWordSpec extends WordSpec with BeforeAndAfterAll {

  var sparkContext: SparkContext = null

  override def beforeAll() = {
    sparkContext = TestingSparkContext.sparkContext
  }

  /**
   * Clean up after the test is done
   */
  override def afterAll() = {
    TestingSparkContext.cleanUp()
    sparkContext = null
  }

  /** assertion that two doubles are almost equal */
  def assertAlmostEqual(x: Double, y: Double, tolerance: Double = 1e-6): Unit = {
    assert(Math.abs(x - y) < tolerance, s"${x} should equal ${y}+-${tolerance}")
  }

}
