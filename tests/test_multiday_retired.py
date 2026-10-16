import importlib.util


def test_multiday_modules_gone():
    assert importlib.util.find_spec('multiphy.multiday') is None
    assert importlib.util.find_spec('multiphy.multiday_runner') is None


def test_backend_job_module_gone():
    assert importlib.util.find_spec('multiphy.backend_job') is None
