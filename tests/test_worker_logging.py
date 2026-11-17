"""A spawned worker logs INFO records to the console, as the controller does.

The test runs a real interpreter that starts a spawn-context process pool with
`_worker_init` as its initializer, as ProcessParallelController does with
max_tasks_per_child, and has a worker log one INFO record.
"""

import subprocess
import sys
import unittest

SCRIPT = """
import logging, multiprocessing, os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OPENAI_API_KEY", "test")
from openevolve.config import Config
from openevolve.database import ProgramDatabase
from openevolve.process_parallel import ProcessParallelController, _worker_init

if __name__ == "__main__":
    config = Config()
    config.logger_prefix = "run1"
    controller = ProcessParallelController(config, "evaluator.py", ProgramDatabase(config.database))
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_worker_init,
        initargs=(controller._serialize_config(config), "evaluator.py", None),
    ) as pool:
        pool.submit(logging.getLogger("openevolve.llm.codex_app_server").info, "hello").result()
"""


class TestWorkerLogging(unittest.TestCase):
    def test_a_spawned_worker_logs_info_with_the_prefixed_name(self):
        result = subprocess.run(
            [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=120
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run1.openevolve.llm.codex_app_server - INFO - hello", result.stderr)


if __name__ == "__main__":
    unittest.main()
