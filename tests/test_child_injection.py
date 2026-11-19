"""A ChildInjector hands the loop programs made outside it (openevolve.injection).

Each test is a real OpenEvolve run: its worker process evaluates the injected child
with an evaluate() that logs its child_context, and the controller places it.
The model is the fake codex app-server, which an injected iteration never calls.
"""

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from openevolve import OpenEvolve
from openevolve.config import Config, LLMModelConfig
from openevolve.injection import ChildInjector, InjectedChild
from openevolve.population import PopulationStrategy

FAKE = Path(__file__).parent / "fake_codex_app_server.py"
INITIAL_PROGRAM = "def f():\n    return 0\n"
INJECTED = "def f():\n    return 2\n"
# Logs each child_context (null for a program no iteration made) to contexts.jsonl
# beside it, and scores a program by its length.
EVALUATOR = """
import dataclasses, json, pathlib
def evaluate(program_path, child_context):
    log = pathlib.Path(__file__).with_name("contexts.jsonl")
    with log.open("a") as f:
        context = None if child_context is None else dataclasses.asdict(child_context)
        f.write(json.dumps(context) + "\\n")
    return {"combined_score": len(pathlib.Path(program_path).read_text()) / 100}
"""


class Inbox:
    """An injector's two calls, recording what they were given."""

    def __init__(self, children):
        self.waiting = list(children)
        self.taken = []
        self.done = []

    def take(self, iteration):
        self.taken.append(iteration)
        return self.waiting.pop(0) if self.waiting else None

    def finish(self, child, admitted):
        self.done.append((child, admitted))


class TestChildInjection(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        (self.tmp / "initial_program.py").write_text(INITIAL_PROGRAM)
        (self.tmp / "evaluator.py").write_text(EVALUATOR)

    def run_with(self, inbox, iterations=1, population_strategy=None):
        """One run of two islands with `inbox` as its injector."""
        config = Config()
        config.evaluator.parallel_evaluations = 1
        config.database.num_islands = 2
        config.database.in_memory = True
        command = [sys.executable, str(FAKE), str(self.tmp / "turns.jsonl"), "0", "x"]
        config.llm.models = [
            LLMModelConfig(
                name="gpt-test",
                provider="codex_app_server",
                timeout=30,
                retries=0,
                retry_delay=0,
                codex_command=command,
            )
        ]
        config.llm.evaluator_models = config.llm.models
        run = OpenEvolve(
            initial_program_path=str(self.tmp / "initial_program.py"),
            evaluation_file=str(self.tmp / "evaluator.py"),
            config=config,
            output_dir=str(self.tmp / "out"),
            population_strategy=population_strategy,
            child_injector=ChildInjector(take=inbox.take, done=inbox.finish),
        )
        asyncio.run(run.run(iterations=iterations))
        return run

    def contexts(self):
        log = self.tmp / "contexts.jsonl"
        return [json.loads(line) for line in log.read_text().splitlines()]

    def test_an_injected_program_is_the_childs_on_its_island_without_a_generation(self):
        child = InjectedChild(code=INJECTED, island=1, label="session-7")
        inbox = Inbox([child])

        run = self.run_with(inbox)

        [stored] = [p for p in run.database.programs.values() if p.parent_id]
        self.assertEqual(stored.code, INJECTED)
        self.assertEqual(stored.metadata["injected"], "session-7")
        self.assertEqual(stored.metadata["evolution_mode"], "injected")
        self.assertIn(stored.id, run.database.islands[1])
        self.assertNotIn(stored.id, run.database.islands[0])
        seed, context = self.contexts()
        self.assertIsNone(seed)
        self.assertEqual(
            context,
            {
                "program_id": stored.id,
                "parent_id": stored.parent_id,
                "island": 1,
                "llm_message": None,
                "injected": "session-7",
                "llm_backend": None,
            },
        )
        self.assertFalse((self.tmp / "turns.jsonl").exists())
        self.assertEqual(inbox.done, [(child, True)])
        self.assertEqual(inbox.taken, [1])

    def test_a_rejected_child_is_done_and_not_admitted(self):
        child = InjectedChild(code=INJECTED, island=0, label="session-8")
        inbox = Inbox([child])
        refuse_children = PopulationStrategy(
            admit=lambda snapshot, candidate, island: candidate.parent_id is None
        )

        run = self.run_with(inbox, population_strategy=refuse_children)

        self.assertEqual(inbox.done, [(child, False)])
        self.assertEqual([p.parent_id for p in run.database.programs.values()], [None])

    def test_an_island_out_of_range_is_refused(self):
        inbox = Inbox([InjectedChild(code=INJECTED, island=2, label="session-9")])

        with self.assertRaisesRegex(ValueError, "child_injector must name an island"):
            self.run_with(inbox)


if __name__ == "__main__":
    unittest.main()
