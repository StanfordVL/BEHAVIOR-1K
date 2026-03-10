def pytest_unconfigure(config):
    import omnigibson as og

    og.shutdown()
