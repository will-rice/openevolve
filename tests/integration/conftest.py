"""
Pytest fixtures for integration tests with optillm server
"""

import os
import shutil
import subprocess

# Import our test utilities
import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))
from test_utils import (
    TEST_MODEL,
    get_evolution_test_evaluator,
    get_evolution_test_program,
    get_integration_config,
    get_test_client,
    is_server_running,
    start_test_server,
    stop_test_server,
)


@pytest.fixture(scope="session")
def optillm_server():
    """Start optillm server for the test session, or reuse one on port 8000 (as CI starts it)

    optillm loads the model on its first request, outside its cache lock
    (optillm/inference.py get_or_load_model), so concurrent first requests load it
    in parallel threads and one copy yields NaN logits: a CUDA device-side assert
    that fails every later request. One request loads it before the tests send
    their parallel ones."""
    proc = None
    port = 8000
    if is_server_running(port):
        print("Using existing optillm server at localhost:8000")
    else:
        print("Starting optillm server for integration tests...")
        proc, port = start_test_server()
        print(f"optillm server started successfully on port {port}")
    try:
        get_test_client(f"http://localhost:{port}/v1").chat.completions.create(
            model=TEST_MODEL, messages=[{"role": "user", "content": "hi"}], max_tokens=1
        )
        yield {"proc": proc, "port": port}
    finally:
        if proc:
            print("Stopping optillm server...")
            stop_test_server(proc)
            print("optillm server stopped")


@pytest.fixture
def evolution_config(optillm_server):
    """Get config for evolution tests"""
    port = optillm_server["port"]
    return get_integration_config(port)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_program_file(temp_workspace):
    """Create a test program file"""
    program_file = temp_workspace / "test_program.py"
    program_file.write_text(get_evolution_test_program())
    return program_file


@pytest.fixture
def test_evaluator_file(temp_workspace):
    """Create a test evaluator file"""
    evaluator_file = temp_workspace / "evaluator.py"
    evaluator_file.write_text(get_evolution_test_evaluator())
    return evaluator_file


@pytest.fixture
def evolution_output_dir(temp_workspace):
    """Create output directory for evolution tests"""
    output_dir = temp_workspace / "output"
    output_dir.mkdir()
    return output_dir
