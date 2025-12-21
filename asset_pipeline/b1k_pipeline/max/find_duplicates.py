import numpy as np
import pymxs
from scipy.spatial.distance import pdist, squareform

rt = pymxs.runtime


def find_duplicates():
    keys = [x for x in rt.objects if rt.classOf(x) == rt.Editable_Poly]
    positions = [
        np.mean(
            np.array(
                rt.polyop.getVerts(
                    obj, rt.execute("#{1..%d}" % rt.polyop.getNumVerts(obj))
                )
            ),
            axis=0,
        )
        for obj in keys
    ]

    positions = np.asarray(positions)
    print(positions.shape)
    distances = squareform(pdist(positions))

    threshold = 1  # mm
    above_threshold = distances < threshold

    candidate_pairs = set()
    for i, key in enumerate(keys):
        if not rt.isValidNode(key):
            continue
        candidates = [
            keys[a]
            for a in np.nonzero(above_threshold[i])[0]
            if rt.isValidNode(keys[a])
        ]
        if len(candidates) > 1:
            pair = tuple({x.baseObject for x in candidates})
            assert len(pair) == 2, (
                "Whoa! " + str(candidates) + " has weird pair " + str(pair)
            )
            if len(pair) == 2:
                candidate_pairs.add(pair)

    print(f"{len(candidate_pairs)} pairs remaining.")
    picked_pair = next(iter(candidate_pairs))
    all_2nd = [x for x in rt.objects if x.baseObject == picked_pair[1]]
    rt.select(all_2nd)


if __name__ == "__main__":
    find_duplicates()
