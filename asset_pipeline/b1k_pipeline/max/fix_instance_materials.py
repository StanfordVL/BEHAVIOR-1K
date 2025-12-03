import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import random
import re
import string
from collections import Counter, defaultdict

import pymxs

import b1k_pipeline.utils

rt = pymxs.runtime


def fix_instance_materials():
    objs_by_model = defaultdict(list)

    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue
        objs_by_model[obj.baseObject].append(obj)

    for objs in objs_by_model.values():
        obj_names = ",".join([x.name for x in objs])
        assert (
            len(
                {
                    (
                        b1k_pipeline.utils.parse_name(obj.name).group("category"),
                        b1k_pipeline.utils.parse_name(obj.name).group("model_id"),
                    )
                    for obj in objs
                }
            )
            == 1
        ), f"More than one cat/model found in {obj_names}."
        count = Counter([x.material for x in objs])
        if len(count) == 1:
            continue
        elif len(count) > 2:
            print(
                f"More than two materials found for instance group including {obj_names}"
            )
            continue

        if len(objs) == 2:
            obj1, obj2 = objs
            mtl1 = obj1.material
            mtl2 = obj2.material
            if (rt.classOf(mtl1) == rt.Multimaterial) == (
                rt.classOf(mtl2) == rt.Multimaterial
            ):
                # Either both are multi-material or neither.
                print(
                    f"Cannot decide which material is right between 2 objects: {obj_names}. Do it manually."
                )
                continue
            elif rt.classOf(mtl1) == rt.Multimaterial:
                # mtl1 is multi-material, mtl2 is not.
                target_mtl = mtl1
            else:
                # mtl2 is multi-material, mtl1 is not.
                target_mtl = mtl2
        else:
            target_mtl, least_seen_count = count.most_common()[-1]
            assert (
                least_seen_count == 1
            ), f"More than one object with least common material found for instance group including {obj_names}."

        for obj in objs:
            obj.material = target_mtl
            print("Fixed", obj.name)


def fix_instance_materials_button():
    try:
        fix_instance_materials()
    except AssertionError as e:
        # Print message
        rt.messageBox(str(e))
        return

    # Print message
    rt.messageBox("Instance materials fixed!")


if __name__ == "__main__":
    fix_instance_materials_button()
