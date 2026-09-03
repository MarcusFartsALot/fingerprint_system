"""Tests for the downloadable attendance scan report."""

from __future__ import annotations

import unittest

import numpy as np

from scan_report import build_scan_analytics_pdf


class ScanReportTests(unittest.TestCase):
    def test_report_is_a_non_empty_pdf(self) -> None:
        analytics = {
            "confusion_matrix": np.array([[3100, 180], [240, 2900]]),
            "metrics": {"Accuracy": 0.94, "F1": 0.93, "Precision": 0.94, "Recall": 0.92},
            "histogram_centres": np.arange(4),
            "prepared_histogram": np.array([10, 20, 30, 40]),
            "enhanced_histogram": np.array([45, 5, 5, 45]),
            "roi_pixels": 6420,
        }

        report = build_scan_analytics_pdf(
            analytics,
            {"Session": "TEST 2", "Student ID": "TEST1"},
            1250.0,
        )

        self.assertTrue(report.startswith(b"%PDF"))
        self.assertGreater(len(report), 3000)


if __name__ == "__main__":
    unittest.main()
