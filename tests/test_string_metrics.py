"""
A metric may be a string, such as a hash that names what a program built.

Program.metrics and EvaluationResult.metrics are Dict[str, Union[float, str]], the
contract the runtime already kept: format_metrics_safe formats string values and
safe_numeric_average skips them. Fitness still comes from numeric metrics, and a
MAP-Elites feature dimension must be numeric to bin.
"""

import os
import tempfile
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase

HASH = "ab" * 32


class TestStringMetrics(unittest.TestCase):
    def setUp(self):
        config = Config()
        config.database.in_memory = True
        config.database.feature_dimensions = ["size"]
        self.db = ProgramDatabase(config.database)

    def test_string_metric_survives_ranking_and_a_checkpoint(self):
        """A better program replaces the best, and a checkpoint keeps the string."""
        for number, score in enumerate((0.25, 0.5)):
            self.db.add(
                Program(
                    id=f"p{number}",
                    code=f"x = {number}",
                    metrics={"combined_score": score, "size": float(number), "hash": HASH},
                )
            )
        self.assertEqual(self.db.get_best_program().id, "p1")
        with tempfile.TemporaryDirectory() as directory:
            self.db.save(directory, iteration=1)
            loaded = ProgramDatabase(self.db.config)
            loaded.load(directory)
        self.assertEqual(loaded.get("p1").metrics["hash"], HASH)

    def test_string_feature_dimension_is_refused(self):
        """A string cannot be binned; the error names the dimension."""
        program = Program(id="p", code="x = 1", metrics={"combined_score": 1.0, "size": HASH})
        with self.assertRaisesRegex(ValueError, "Feature dimension 'size' must be numeric"):
            self.db.add(program)


if __name__ == "__main__":
    unittest.main()
