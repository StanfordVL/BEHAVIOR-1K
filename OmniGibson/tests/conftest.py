def pytest_addoption(parser):
    parser.addoption("--test-args", action="store", default="", help="Extra args passed to the example under test")


def pytest_unconfigure(config):
    import omnigibson as og

    og.shutdown()
