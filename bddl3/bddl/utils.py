########## CUSTOM ERRORS ############


class UncontrolledCategoryError(Exception):
    def __init__(self, malformed_cat):
        self.malformed_cat = malformed_cat


class UnsupportedPredicateError(Exception):
    def __init__(self, predicate):
        self.predicate = predicate
