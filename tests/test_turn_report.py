"""TurnReport's failure reasons and the error text each example keeps."""

import unittest

from openevolve.llm.turn_report import ERROR_CHARS, TurnReport, failure_reason

# A shell call's output as codex sends it: a long file print, then the refusal.
PRINTED = "\n".join(f"line {i} of a printed program" for i in range(40))
REFUSED = "touch: cannot touch '/usr/qcc_probe': Read-only file system"


class TestFailureReasons(unittest.TestCase):
    """A shell's refusal and a missing path are told apart, wherever they print."""

    def test_a_refusal_after_long_output_is_sandbox_and_kept(self):
        error = f"{PRINTED}\n{REFUSED}\n"
        self.assertGreater(error.index(REFUSED), ERROR_CHARS)
        report = TurnReport(agent="codex", backend="openai/gpt-test")
        report.add_tool_failure(failure_reason(True, error), "commandExecution", "c", error)
        [example] = report.tool_failures["sandbox"].examples
        self.assertEqual(example.error, REFUSED)

    def test_a_missing_path_is_not_a_refusal(self):
        error = "ls: cannot access 'built/invsqrt.kmb': No such file or directory\n"
        self.assertEqual(failure_reason(True, error), "missing_file")
        self.assertEqual(failure_reason(False, error), "other")


if __name__ == "__main__":
    unittest.main()
