"""A hook that hands the evolution loop programs made outside it.

Some programs are written by something other than the loop's LLM calls: a long
offline session, a person. A ``ChildInjector`` lets the loop take one in place of
an iteration's generation: before each iteration is submitted, the controller asks
``take`` for a waiting program; when it gets one, that iteration skips the prompt
and the LLM, and the program is its child, on the island the injector names. It
then goes the way every child goes: evaluation (whose ``child_context`` says it was
injected, by ``label``), the population strategy's admission, MAP-Elites placement
and checkpoints. Once the iteration has been processed, however it ended, the
controller calls ``done`` with whether the database admitted the child.

``take`` and ``done`` run in the controller's process, on its event loop's thread.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class InjectedChild:
    """A program to make an iteration's child, on an island.

    Attributes:
        code: The program.
        island: The island it joins; its parent is drawn from this island.
        label: What the injector calls it; evaluate() gets it as
            ``ChildContext.injected``, and the child's metadata as ``injected``.
    """

    code: str
    island: int
    label: str


@dataclass(frozen=True)
class ChildInjector:
    """Where waiting programs come from, and what to tell when one was processed.

    Attributes:
        take: Given the iteration about to be submitted, a waiting program, or
            None to let the iteration generate as usual. It should not hand out a
            program again until ``done`` was called for it.
        done: Called once with the child and whether the database admitted it,
            after its iteration was processed: admitted, rejected, failed or
            timed out.
    """

    take: Callable[[int], Optional[InjectedChild]]
    done: Callable[[InjectedChild, bool], None]
