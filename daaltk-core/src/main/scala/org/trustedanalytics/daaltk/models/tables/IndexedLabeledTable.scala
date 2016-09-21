package org.trustedanalytics.daaltk.models.tables

/**
 * Indexed numeric tables with features and corresponding labels
 *
 * @param features Numeric table with features
 * @param labels Numeric table with labels
 */
case class IndexedLabeledTable(features: IndexedNumericTable, labels: IndexedNumericTable)
