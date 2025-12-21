from collections import defaultdict
import sys

import tqdm

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import pymxs

rt = pymxs.runtime

from b1k_pipeline.utils import parse_name


def match_links_between_instances(obj):
    # Parse the name and assert it's the base link of a non-bad object
    parsed_name = parse_name(obj.name)
    assert parsed_name, f"Object {obj.name} has no parsed name"
    # assert not parsed_name.group("bad"), f"Object {obj.name} is a bad object"
    assert not (
        parsed_name.group("link_name") and parsed_name.group("link_name") != "base_link"
    ), f"Object {obj.name} is not the base link"

    # Get the model ID
    loose = parsed_name.group("loose")
    if not loose:
        loose = ""
    model_id = parsed_name.group("model_id")
    instance_id = parsed_name.group("instance_id")

    # First, find all the parts of this object's model, across all of its instances.
    all_same_model_id = [
        x
        for x in rt.objects
        if (
            parse_name(x.name)
            and parse_name(x.name).group("model_id") == model_id
            and parse_name(x.name).group("joint_side") != "upper"
            and not parse_name(x.name).group("meta_type")
            and rt.classOf(x) == rt.Editable_Poly
        )
    ]
    same_model_id_parsed_names = [parse_name(x.name) for x in all_same_model_id]
    assert all(
        same_model_id_parsed_names
    ), "All same model ID objects should have a parsed name"

    # Check that they are all marked as non BAD
    # assert all(
    #     not x.group("bad") for x in same_model_id_parsed_names
    # ), "All instances of the same model ID should be marked as non-bad."

    # Check that once grouped by instance ID, they all have the same set of links
    links_by_instance_id = defaultdict(dict)
    for inst, inst_parsed_name in zip(all_same_model_id, same_model_id_parsed_names):
        link_name = inst_parsed_name.group("link_name")
        if not link_name:
            link_name = "base_link"
        links_by_instance_id[inst_parsed_name.group("instance_id")][link_name] = inst

    # Unroll to find all links
    all_links = frozenset(
        link
        for instance_links_to_objs in links_by_instance_id.values()
        for link in instance_links_to_objs
    )
    print("Found links", sorted(all_links))

    if all_links == {"base_link"}:
        print("No links to match")
        return

    # Check that the currently selected instance contains all the links
    base_instance_id = instance_id
    if frozenset(links_by_instance_id[base_instance_id].keys()) != all_links:
        # Find a maximal instance as the base instance if the current one isnt
        for inst_id, links_to_objs in links_by_instance_id.items():
            if frozenset(links_to_objs.keys()) == all_links:
                base_instance_id = inst_id
                break
    base_obj = links_by_instance_id[base_instance_id]["base_link"]
    assert (
        frozenset(links_by_instance_id[base_instance_id].keys()) == all_links
    ), f"Instance {base_obj.name} does not contain all links some other instances have. Pick the maximal instance."

    # For this instance, store the relative transforms of all the links
    relative_transforms = {}
    inverse_base_link_transform = rt.inverse(base_obj.transform)
    base_link_offset_scale = base_obj.objectoffsetscale
    for link_name, link_obj in links_by_instance_id[base_instance_id].items():
        if link_name == "base_link":
            continue
        relative_transforms[link_name] = (
            link_obj.transform * inverse_base_link_transform
        )
        # print(f"Relative transform for {link_name} is {relative_transforms[link_name]}")

    # Walk through the instances to find the links that are not present in all instances
    for inst_id, links_to_objs in links_by_instance_id.items():
        if inst_id == base_instance_id:
            continue

        # Assert that the base link exists and grab it
        assert (
            "base_link" in links_to_objs
        ), f"Instance {inst_id} does not have a base link"
        instance_base_link = links_to_objs["base_link"]
        instance_base_link_transform = instance_base_link.transform
        instance_base_link_offset_scale = instance_base_link.objectoffsetscale

        # Iterate through all the links, copy missing ones, and match transforms.
        for link_name in all_links:
            if link_name == "base_link":
                continue
            link_obj = links_to_objs.get(link_name, None)
            corresponding_link_on_base_instance = links_by_instance_id[
                base_instance_id
            ][link_name]
            if link_obj is None:
                print(f"Copying link {link_name} for instance {inst_id}")
                copy_name = corresponding_link_on_base_instance.name.replace(
                    f"-{base_instance_id}-", f"-{inst_id}-"
                )
                success, copies = rt.maxOps.cloneNodes(
                    [corresponding_link_on_base_instance],
                    cloneType=rt.name("instance"),
                    newNodes=pymxs.byref(None),
                )
                assert success, f"Could not clone {link_name} for instance {inst_id}"
                (link_obj,) = copies
                link_obj.name = copy_name
            link_obj.transform = (
                relative_transforms[link_name] * instance_base_link_transform
            )
            scale_ratio = instance_base_link_offset_scale / base_link_offset_scale
            link_obj.objectoffsetscale = (
                corresponding_link_on_base_instance.objectoffsetscale * scale_ratio
            )
            link_obj.objectoffsetpos = (
                corresponding_link_on_base_instance.objectoffsetpos * scale_ratio
            )


def process_all_objects():
    # Get all non-bad, zeroth-instance, base-link editable polies and process them
    MODEL_WHITELIST = {
        "bamfsz",
        "bkdxqa",
        "bmsclc",
        "ceddpg",
        "dladgw",
        "dmwxyl",
        "dnyzym",
        "dqnbsj",
        "ehwmol",
        "emeeke",
        "eobsmt",
        "fexqbj",
        "fqhdne",
        "fqvgny",
        "fwstpx",
        "gemgfz",
        "gfrgkk",
        "ggcyib",
        "gjrero",
        "gpkbiw",
        "ijnwlp",
        "immwzb",
        "inmymj",
        "ithrgo",
        "jhymlr",
        "jongdb",
        "jrhgeu",
        "kctcuj",
        "kfmkbm",
        "ktydvs",
        "kubcdk",
        "kwbnhy",
        "lcrefl",
        "lsyzkh",
        "lvgliq",
        "lwjdmj",
        "mfocvp",
        "nfrbch",
        "nnvyol",
        "ntlgsx",
        "nwlbit",
        "ohagsq",
        "pluwfl",
        "pwoerr",
        "pxhbim",
        "qohxjq",
        "rgmujm",
        "rsypfh",
        "rvunhj",
        "rxvopf",
        "slgzfc",
        "spojpj",
        "taqzxq",
        "ufhpbn",
        "ulnafj",
        "uztisk",
        "vudhlc",
        "wwhydr",
        "xbfgjc",
        "xiajyb",
        "xxipyh",
        "xyejdx",
        "yfycfx",
        "ykfcaz",
    }
    for obj in rt.objects:
        if rt.classOf(obj) != rt.Editable_Poly:
            continue
        parsed_name = parse_name(obj.name)
        if not parsed_name:
            continue
        if parsed_name.group("model_id") not in MODEL_WHITELIST:
            continue
        if parsed_name.group("instance_id") != "0":
            continue
        if (
            parsed_name.group("link_name")
            and parsed_name.group("link_name") != "base_link"
        ):
            continue
        if parsed_name.group("bad"):
            continue
        if parsed_name.group("meta_type"):
            continue
        match_links_between_instances(obj)


def main():
    match_links_between_instances(rt.selection[0])


if __name__ == "__main__":
    main()
