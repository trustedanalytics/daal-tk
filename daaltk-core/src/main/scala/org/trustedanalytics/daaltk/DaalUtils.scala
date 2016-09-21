package org.trustedanalytics.daaltk

import java.io.File
import com.intel.daal.services.DaalContext
import org.apache.commons.lang.StringUtils
import com.typesafe.config.{ ConfigFactory }

import scala.util.Try

object DaalUtils extends Serializable {

  lazy val config = ConfigFactory.load(this.getClass.getClassLoader)

  val DaalRequiredLibraries = List("libDaalTkJavaAPI.so", "libiomp5.so", "libJavaAPI.so", "/libtbb.so.2")

  /**
   * Path to DAAL dynamic libraries
   */
  val daalDynamicLibraries: String = Try(config.getString("trustedanalytics.daaltk.daal.dynamic-libraries")).getOrElse("")

  /**
   * Check that all required libraries are specified in configuration.
   *
   * Throws an exception if any of the required libraries are not specified in configuration
   *
   * @param confLibraryPath String with comma-separated list of DAAL libraries in configuration
   */
  def validateDaalLibraries(confLibraryPath: String): Unit = {
    require(StringUtils.isNotEmpty(confLibraryPath), "trustedanalytics.daaltk.daal.dynamic-libraries cannot be empty. " + "" +
      "Please set path to DAAL libraries in configuration file.")
    val libraryFiles = confLibraryPath.split(",")

    DaalRequiredLibraries.foreach(library => {
      val pattern = s".*(${library})".r
      if (!libraryFiles.exists(f => pattern.findFirstIn(f).isDefined)) {
        throw new scala.IllegalArgumentException(s"Please add path to ${library} in trustedanalytics.daaltk.daal.dynamic-libraries.")
      }
    })
  }

  /**
   * Get list of files to DAAL libraries
   *
   * @param confLibraryPath  String with comma-separated list of DAAL libraries in configuration
   * @return List of files to DAAL libraries
   */
  def getDaalLibraryPaths(confLibraryPath: String): List[File] = {
    validateDaalLibraries(confLibraryPath)
    val paths = confLibraryPath.split(",").toList
    paths.map(p => new File(p))
  }

  /**
   * Interface for DAAL context results
   * @tparam T Return type for operation on DAAL context
   */
  trait DaalContextResult[T] {

    /**
     * Return result of type T or throw exception with provided error message
     * @param message Error message to display in case of exception
     * @return Result of type T or exception in case of error
     */
    def elseError(message: String): T
  }

  /**
   * Return type for successful operations with DAAL context
   * @param result Successful result
   */
  class DaalContextSuccessResult[T](result: T) extends DaalContextResult[T] {

    /**
     * Return valid result on successful operation
     * @param message Error message to display in case of exception
     * @return Valid result
     */
    override def elseError(message: String): T = result
  }

  /**
   * Return type for failed operations with DAAL context
   * @param ex Exception thrown
   */
  class DaalContextErrorResult[T](ex: Exception) extends DaalContextResult[T] {

    /**
     * Throw exception on failed operation
     * @param message Error message to display
     * @return Exception thrown
     */
    override def elseError(message: String): T = {
      throw new RuntimeException(message, ex)
    }
  }

  /**
   * Execute a code block using DAAL context
   *
   * @param func Function which uses DAAL context
   * @return Valid result of type T, or exception in case of error
   */
  def withDaalContext[T](func: DaalContext => T): DaalContextResult[T] = {
    val context = new DaalContext()
    try {
      new DaalContextSuccessResult[T](func(context))
    }
    catch {
      case ex: Exception =>
        new DaalContextErrorResult[T](ex)
    }
    finally {
      context.dispose()
    }
  }

}
