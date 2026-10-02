"""Smoke tests for the public command-line entry points."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class CommandLineEntrypointTests(unittest.TestCase):
    def test_all_documented_ml_commands_show_help_without_loading_model_weights(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        modules = (
            "scripts.train_retriever",
            "scripts.train_reranker",
            "scripts.generate_rationales",
            "scripts.evaluate",
            "scripts.build_report",
        )

        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertTrue(result.stdout.startswith("usage:"), result.stdout[:200])


if __name__ == "__main__":
    unittest.main()
