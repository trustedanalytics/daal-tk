package org.trustedanalytics.daaltk.models.tables

import com.intel.daal.data_management.data.NumericTable
import scala.language.implicitConversions

/**
 * These implicits can be imported to add conversion functions related functions to DAAL tables
 */
object DaalConversionImplicits {
  implicit def numericTableFunctions(self: NumericTable): DaalNumericTableFunctions = new DaalNumericTableFunctions(self)
}
