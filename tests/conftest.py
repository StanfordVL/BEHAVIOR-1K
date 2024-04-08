import omnigibson as og


def pytest_unconfigure(config):
    og.shutdown()
