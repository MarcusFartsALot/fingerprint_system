"""Tests for the dependency-free attendance scan analytics."""

from __future__ import annotations

import unittest

import numpy as np

from scan_analytics import analyse_scan


class ScanAnalyticsTests(unittest.TestCase):
    def test_metrics_and_histograms_are_bounded(self) -> None:
        prepared = np.full((80, 80), 210, dtype=np.uint8)
        prepared[:, 10:14] = 35
        prepared[:, 30:34] = 35
        prepared[:, 50:54] = 35
        denoised = prepared.copy()
        enhanced = np.full_like(prepared, 255)
        enhanced[:, 10:14] = 0
        enhanced[:, 30:34] = 0
        enhanced[:, 50:54] = 0
        mask = np.full_like(prepared, 255)

        analytics = analyse_scan(prepared, enhanced, mask)

        self.assertEqual(analytics["confusion_matrix"].shape, (2, 2))
        self.assertEqual(int(analytics["confusion_matrix"].sum()), 80 * 80)
        self.assertEqual(len(analytics["histogram_centres"]), 32)
        self.assertEqual(int(analytics["prepared_histogram"].sum()), 80 * 80)
        self.assertEqual(int(analytics["enhanced_histogram"].sum()), 80 * 80)
        for score in analytics["metrics"].values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_empty_region_is_rejected(self) -> None:
        image = np.zeros((40, 40), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "non-empty fingerprint region"):
            analyse_scan(image, image, image)


if __name__ == "__main__":
    unittest.main()
