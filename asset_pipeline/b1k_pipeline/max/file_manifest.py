import hashlib
import json
import pathlib
import pymxs
from pymxs import runtime as rt
from tqdm import tqdm

import sys

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

from b1k_pipeline.max.prebake_textures import hash_object

OUTPUT_FILENAME = "file_manifest.json"
OUTPUT_FILENAME_DEEP = "file_manifest_deep.json"


class MaxscriptEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) == pymxs.MXSWrapperBase:
            # Here we support certain known types
            if rt.classOf(obj) == rt.Color:
                return [obj.r, obj.g, obj.b, obj.a]
            elif rt.classOf(obj) == rt.Array:
                return list(obj)
            elif rt.classOf(obj) == rt.Time:
                return obj.frame
        # Let the base class default method raise the TypeError
        return super().default(obj)


def hash_single_thing(thing):
    # try:
    # This tries to check if the thing can be JSON dumped. If it can, then we can do a proper hash
    # by sha256'ing its bytes.
    json_string = json.dumps(
        thing, sort_keys=True, separators=(",", ":"), cls=MaxscriptEncoder
    )

    # Encode the string to bytes, as hashlib works on bytes
    json_bytes = json_string.encode("utf-8")

    # Use a standard hash algorithm like SHA-256
    # Use hexdigest() for a string representation of the hash
    return hashlib.sha256(json_bytes).hexdigest()
    # except TypeError:
    #     # Otherwise, we have to rely on the hash() function which is not stable across runs for
    #     # non-maxscript objects. But - no alternatives here!
    #     return hash(thing)


def should_skip_attr(k, v):
    # We skip None because its hash changes every time we run the script.
    # It's fine to do this because if the value changes to not none, it will stop being
    # skipped and we'll get a different hash.
    if v is None:
        return True
    if k in [
        "setmxsprop",
        "getmxsprop",
        "bitmap",
        "excludeList",
        "brdf_newGTRAnisotropy",
    ]:
        return True
    # These get hashed as part of the submtl / subtexmap searches.
    if rt.superClassOf(v) == rt.textureMap or rt.superClassOf(v) == rt.Material:
        return True
    return False


def hash_attrs(obj):
    hash_dict = {}
    for k in dir(obj):
        try:
            v = getattr(obj, k)
            if not should_skip_attr(k, v):
                hash_dict[k] = hash_single_thing(v)
        except:
            hash_dict[k] = None

    # Here it's possible to just return the dict rather than hashing it again. That allows saving
    # far more detailed information e.g. as to what was changed about the object or material. However
    # it makes the files grow to tens of megabytes and makes them really hard to diff so we don't do that.
    # But if you want to, just return hash_dict instead of the hash to get the full information for
    # debugging.
    return hash_dict  # {"unified_hash": hash_single_thing(hash_dict)}


def hash_material(root_mat):
    """
    Get the hash of a material by recursively hashing its sub-materials.
    This is a very blind hash - it will just identify if ANYTHING in the hierarchy
    of materials has changed.
    """

    hash_dict = {}

    def _recursively_hash_materials_and_textures(mtl, partial_hash_dict):
        # Do the actual hashing
        partial_hash_dict.update(hash_attrs(mtl))

        # We can check for submtls for material instances
        if rt.superClassOf(mtl) == rt.Material:
            partial_hash_dict["submtls"] = {}
            for i in range(rt.getNumSubMtls(mtl)):
                sub_mtl = rt.getSubMtl(mtl, i + 1)
                sub_mtl_slot_name = rt.getSubMtlSlotName(mtl, i + 1)
                partial_hash_dict["submtls"][sub_mtl_slot_name] = {}
                if sub_mtl is not None:
                    _recursively_hash_materials_and_textures(
                        sub_mtl, partial_hash_dict["submtls"][sub_mtl_slot_name]
                    )

        # We can check for subtexmaps for texture maps and materials
        if rt.superClassOf(mtl) == rt.textureMap or rt.superClassOf(mtl) == rt.Material:
            partial_hash_dict["subtexmaps"] = {}
            for i in range(rt.getNumSubTexmaps(mtl)):
                sub_texmap = rt.getSubTexmap(mtl, i + 1)
                sub_texmap_slot_name = rt.getSubTexmapSlotName(mtl, i + 1)
                partial_hash_dict["subtexmaps"][sub_texmap_slot_name] = {}
                if sub_texmap is not None:
                    _recursively_hash_materials_and_textures(
                        sub_texmap,
                        partial_hash_dict["subtexmaps"][sub_texmap_slot_name],
                    )

    _recursively_hash_materials_and_textures(root_mat, hash_dict)
    return hash_dict


def main():
    # Go through all the objects and store their information.
    file_manifest = []
    file_manifest_deep = []
    for obj in tqdm(list(rt.objects)):
        obj_id = obj.inode.handle
        obj_name = obj.name
        obj_parent = obj.parent.name if obj.parent else None
        obj_layer = obj.layer.name
        obj_class = str(rt.classOf(obj))
        if rt.classOf(obj) == rt.Editable_Poly:
            obj_hash = hash_object(obj)
        else:
            obj_hash = hash_attrs(obj)
        obj_mtl_hash = hash_material(obj.material) if obj.material else None
        obj_transform = str(obj.transform)
        obj_objecttransform = str(obj.objectTransform)
        file_manifest_deep.append(
            {
                "id": obj_id,
                "name": obj_name,
                "parent": obj_parent,
                "layer": obj_layer,
                "class": obj_class,
                "hash": obj_hash,
                "mtl_hash": obj_mtl_hash,
                "transform": obj_transform,
                "objecttransform": obj_objecttransform,
            }
        )
        file_manifest.append(
            {
                "id": obj_id,
                "name": obj_name,
                "parent": obj_parent,
                "layer": obj_layer,
                "class": obj_class,
                "hash": obj_hash
                if isinstance(obj_hash, str)
                else hash_single_thing(obj_hash),
                "mtl_hash": hash_single_thing(obj_mtl_hash),
                "transform": obj_transform,
                "objecttransform": obj_objecttransform,
            }
        )

    # Sort the list by id
    file_manifest_deep.sort(key=lambda x: x["name"])
    file_manifest.sort(key=lambda x: x["name"])

    # Dump into a JSON file
    output_dir = pathlib.Path(rt.maxFilePath) / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / OUTPUT_FILENAME
    with open(filename, "w") as f:
        json.dump(file_manifest, f, indent=4)

    filename_deep = output_dir / OUTPUT_FILENAME_DEEP
    with open(filename_deep, "w") as f:
        json.dump(file_manifest_deep, f, indent=4)


if __name__ == "__main__":
    main()
