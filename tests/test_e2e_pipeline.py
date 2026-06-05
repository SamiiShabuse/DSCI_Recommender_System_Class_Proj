"""End-to-end smoke test for the recommender pipeline.

Run from the repository root:

    python -m unittest discover -s tests
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import PipelineConfig, run_pipeline


class PipelineE2ETest(unittest.TestCase):
    def test_synthetic_pipeline_writes_metrics_and_figures(self) -> None:
        """Run all pipeline stages with built-in demo profiles.

        # IMPORTANT:
        Smoke test only: this verifies the full pipeline can run end-to-end,
        evaluate all five models, and write the expected output files.
        It does not validate exact metric values or model quality.
        """
        temp_root = Path(__file__).resolve().parent / "_tmp_e2e" / uuid.uuid4().hex
        temp_root.mkdir(parents=True)
        try:
            data_dir = temp_root / "empty_data"
            output_dir = temp_root / "artifacts"
            data_dir.mkdir()

            outputs = run_pipeline(
                PipelineConfig(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    target_posts=1000,
                    synthetic=True,
                    k=5,
                    seed=42,
                )
            )

            results_df = outputs["results_df"]

            self.assertEqual(len(outputs["posts_base_df"]), 1000)
            self.assertEqual(
                set(results_df["model"]),
                {
                    "global_baseline",
                    "category_baseline",
                    "user_based_cf",
                    "content_based",
                    "hybrid",
                },
            )
            self.assertGreater(results_df["evaluated_users"].max(), 0)
            self.assertTrue(outputs["results_path"].exists())
            self.assertTrue(outputs["run_summary_path"].exists())
            for figure_path in outputs["figure_paths"]:
                self.assertTrue(figure_path.exists(), figure_path)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
            shutil.rmtree(temp_root.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
