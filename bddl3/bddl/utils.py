"""Custom exception types used throughout the BDDL library."""


class UncontrolledCategoryError(Exception):
    """Raised when a BDDL condition references an object category that is not
    present in the object scope.

    This typically means the problem file mentions an object synset that was not
    declared in the ``:objects`` section, so there is no scope entry for it.

    Args:
        malformed_cat: The category string that could not be resolved.
    """

    def __init__(self, malformed_cat):
        self.malformed_cat = malformed_cat


class UnsupportedPredicateError(Exception):
    """Raised when a BDDL condition references a predicate that is not defined
    in the domain file.

    Args:
        predicate: The predicate name that was not found.
    """

    def __init__(self, predicate):
        self.predicate = predicate
