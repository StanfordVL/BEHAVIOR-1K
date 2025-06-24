###### COMBINATORICS UTILS ######

import itertools
from bddl.config import GROUND_GOALS_MAX_OPTIONS, GROUND_GOALS_MAX_PERMUTATIONS


def truncated_product(*sequences, max_options=GROUND_GOALS_MAX_OPTIONS):
    """Breadth-first search cartesian product 
       Source: https://stackoverflow.com/questions/42288203/generate-itertools-product-in-different-order

    :yields (tuple): elements of cartesian product 
    """
    # sequences = tuple(tuple(seq) for seqin sequences)
    counter = 0

    def partitions(n, k):
        for c in itertools.combinations(range(n + k - 1), k - 1):
            yield (b - a - 1 for a, b in zip((-1,) + c, c + (n + k -1,)))

    max_position = [len(i) - 1 for i in sequences]
    for i in range(sum(max_position)):
        if counter >= max_options:
            break 
        for positions in partitions(i, len(sequences)):
            try:
                if counter < max_options:
                    counter += 1
                    yield tuple(map(lambda seq, pos: seq[pos], sequences, positions))
                else: 
                    break
            except IndexError:
                continue
    # print("next try:", tuple(map(lambda seq, pos: seq[pos], sequences, max_position)))
    if counter < max_options:
        counter += 1
        yield tuple(map(lambda seq, pos: seq[pos], sequences, max_position))


def truncated_permutations(iterable, r=None, max_permutations=GROUND_GOALS_MAX_PERMUTATIONS):
    """Adapted from https://docs.python.org/3/library/itertools.html#itertools.permutations
    """
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r 
    if r > n:
        return 
    indices = list(range(n))
    cycles = list(range(n, n - r, -1))

    counter = 0
    if counter < max_permutations:
        counter += 1
        yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + indices[i : i + 1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                if counter < max_permutations:
                    counter += 1
                    yield tuple(pool[i] for i in indices[:r])
                else:
                    return 
                break
        else:
            return


########## CUSTOM ERRORS ############

class UncontrolledCategoryError(Exception):
    def __init__(self, malformed_cat):
        self.malformed_cat = malformed_cat

class UnsupportedPredicateError(Exception):
    def __init__(self, predicate):
        self.predicate = predicate


if __name__ == "__main__":
    import pprint
    sample_small = [[[['ap1', 'tab1']], [['ap1', 'tab2']]], [[['ap2', 'tab1']], [['ap2', 'tab2']]], [[['ap3', 'tab1']], [['ap3', 'tab2']]]]
    # pprint.pprint(list(truncated_product(*sample_small)))
    # print(list(truncated_permutations(range(10), 3)))
    list(truncated_permutations(range(10), 3))
