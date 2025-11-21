"""
The program takes in UpWorker's .csv survey data, Manual Annotations (Canonical), and GPT Annotations (Canonical) and merges them.
This merged canonical is then run through the Programmatic Annotation system to create a Master Canonical. (Note: Master /= Final)
The Final step for the Master is to be run through propogate_by_intersection.py which has not been completed

Separately, once Master has been compiled, the two hierarchy generation functions will be run. The first function generates a hierarchy of all 
"""


from calendar import c
import json
import pathlib 
from nltk.corpus import wordnet as wn
import bddl.data_generation.prop_config as pcfg
from bddl.bddl_verification import VALID_ATTACHMENTS


CANONICAL_FN = pathlib.Path(__file__).parents[1] / "generated_data" / "syn_prop_annots_canonical.json"
PROP_TO_SYN_FILE = pathlib.Path(__file__).parents[1] / "generated_data" / "properties_to_synsets.json" # gives all applicable synsets for a given property
SYN_TO_DESC_FILE = pathlib.Path(__file__).parents[1] / "generated_data" / "synsets_to_descriptors.json" # gives states for a given synset

DESIRED_PROPS = set(pcfg.CROWD_PROP_NAMES + pcfg.GPT_PROP_NAMES + pcfg.INTERNAL_PROP_NAMES)

PROP_TO_DESC = { #added by hand from B1K Object States Google Doc found here https://docs.google.com/document/d/10g5A-ODF4POFN0SxIhiQIuDh5L1X_8KXriwKfaKWUFc/edit?usp=sharing as of May 12, 2022
    "breakable": "broken",
    "burnable": "burnt",
    "cleaningTool": None,
    "coldSource": None,
    "cookable": "cooked",
    "dustyable": "dusty",
    "freezable": "frozen",
    "heatSource": None,
    "liquid": None,
    "openable": "open",
    "perishable": "perished",
    "screwable": "screwed",
    "stainable": "stained",
    "sliceable": "sliced",
    "slicer": None,
    "diceable": "diced",
    "soakable": "soaked",
    "timeSetable": "time_set",
    "toggleable": "toggled_on",
    "waterSource": None,
    "foldable": "folded",
    "unfoldable": None,     # this property actually didn't even need to be annotated, because it's not used and it's just whatever's `foldable.` Or, empty and closed can be annotated, but similarly not put in desc list.
    "soakable": None,
    "wetable": "wet",
    "flammable": "on_fire",
    "assembleable": None,
    "heatable" : "hot",
    "boilable": "boiling",
    "meltable" : "melted",
    "fireSource": None,
    "overcookable": "overcooked",
    "substance": None,
    "microPhysicalSubstance": None,
    "macroPhysicalSubstance": None,
    "visualSubstance": None,
    "deformable": None,
    "softBody": None,
    "cloth": None,
    "stickyable": "sticky",
    "grassyable": "grassy",
    "scratchable": "scratched",
    "tarnishable": "tarnished",
    "moldyable": "moldy",
    "rustable": "rusty",
    "wrinkleable": "wrinkly",
    "disinfectable": "disinfected",
    "attachable": "attached",
    "mixable": None,
    "blendable": None,
    "rope": None,
    "fillable": None,
    "rigidBody": None,
    "nonSubstance": None,
    "particleRemover": None,
    "particleApplier": None,
    "particleSource": None,
    "particleSink": None,
    "needsOrientation": None,
    "waterCook": None,
    "nonDeformable": None,
    "nonSubstance": None,
    "sceneObject": None,
    "drapeable": None,
    "physicalSubstance": None,
    "mixingTool": None,
}

########### GET INPUT CANONICALS ###########

def get_annots_canonical(syn_prop_dict):
    canonical = {}
    for synset, prop_binaries in syn_prop_dict.items():
        synset_canonical = {}
        for prop, applies in prop_binaries.items():
            if prop in DESIRED_PROPS and bool(applies) and float(applies): 
                synset_canonical[prop] = {}
            elif prop == "objectType" and applies in DESIRED_PROPS:
                synset_canonical[applies] = {}
                if applies not in ["liquid", "visualSubstance", "macroPhysicalSubstance", "microPhysicalSubstance"]:
                    synset_canonical["nonSubstance"] = {}
        canonical[synset] = synset_canonical
    canonical_with_programmatic = add_programmatic_properties(canonical)
    return canonical_with_programmatic


# Removed original property annotation sources - human, GPT-3, and manual. If
#   we go back to those sources, we should write code to get the annotations 
#   into the unified Google sheet (that is then pulled here as a CSV) - better
#   way to handle things.


############ GETTING PROPERTIES #############

def add_programmatic_properties(synset_content): # runs programmatic addition over existing canonical input
    attachable_objects = set()
    for pair in VALID_ATTACHMENTS:
        attachable_objects.add(pair[0])
        attachable_objects.add(pair[1])
    for synset in synset_content:
        if "liquid" in synset_content[synset]:
            synset_content[synset]["boilable"] = {}
        if ("visualSubstance" in synset_content[synset]) or ("microPhysicalSubstance" in synset_content[synset]) or ("macroPhysicalSubstance" in synset_content[synset]) or ("liquid" in synset_content[synset]):
            synset_content[synset]["substance"] = {}
        if ("microPhysicalSubstance" in synset_content[synset]) or ("macroPhysicalSubstance" in synset_content[synset]) or ("liquid" in synset_content[synset]):
            synset_content[synset]["physicalSubstance"] = {}
        if ("cloth" in synset_content[synset]) or ("rope" in synset_content[synset]) or ("softBody" in synset_content[synset]):
            synset_content[synset]["deformable"] = {}
        if ("cloth" in synset_content[synset]) or ("rope" in synset_content[synset]):
            synset_content[synset]["drapeable"] = {}
        if ("nonSubstance" in synset_content[synset]) and ("cookable" in synset_content[synset] or "fillable" in synset_content[synset]):
            # cookables and fillables are both heatable
            synset_content[synset]["heatable"] = {}
        if ("nonSubstance" in synset_content[synset]) and ("cookable" in synset_content[synset] or "fillable" in synset_content[synset]):
            # cookables and fillables are both freezable
            synset_content[synset]["freezable"] = {}
        if synset in attachable_objects:
            synset_content[synset]["attachable"] = {}
        if "nonSubstance" in synset_content[synset]: # non-substances are both wetable and mixable
                synset_content[synset].update({
                    "wetable": {},
                    "stickyable": {},
                    "dustyable": {},
                    "grassyable": {},
                    "scratchable": {},
                    "stainable": {},
                    "tarnishable": {},
                    "moldyable": {},
                    "rustable": {},
                    "wrinkleable": {},
                    "disinfectable": {},
                })


        # Hardcode in a place where the synset is seen, since not all scene synsets made it:
        if synset == "sink.n.01":
            synset_content[synset].update({"waterSource": {}})

    return synset_content


# Get properties_to_synsets.json
def make_properties_to_synsets(final_canonical):
    if type(final_canonical) == str: 
        with open(final_canonical, "r") as f:
            final_canonical = json.load(f)
    properties_to_synsets = {}
    for syn, props in final_canonical.items(): 
        for prop in props: 
            if prop not in properties_to_synsets:
                properties_to_synsets[prop] = []
            properties_to_synsets[prop].append(syn)
    return properties_to_synsets


def get_synset_descriptors(synsets_to_filtered_properties): # take in canonical format
    synsets_to_states = {} # declare an empty dictionary
    for synset_label, properties in synsets_to_filtered_properties.items():
        synsets_to_states[synset_label] = [[PROP_TO_DESC[prop]] * 2 for prop in properties if PROP_TO_DESC[prop] is not None] # if the property has an attached state, add its states
    return synsets_to_states


# API - only these should be used outside these script, and these should only be used outside this script

def create_get_save_annots_canonical(syn_prop_dict):
    print("Creating canonical annots file...")
    canonical = get_annots_canonical(syn_prop_dict)
    with open(CANONICAL_FN, "w") as f:
        json.dump(canonical, f, indent=4)
    print("Created and saved canonical annots file.")
    return canonical

def create_get_save_properties_to_synsets(propagated_canonical):
    print("Creating properties to synsets...")
    props_to_syns = make_properties_to_synsets(propagated_canonical)
    with open(PROP_TO_SYN_FILE, "w") as f:
        json.dump(props_to_syns, f, indent=4)
    print("Created and saved properties to synsets file.")
    return props_to_syns

def create_get_save_synsets_to_descriptors(propagated_canonical):
    print("Creating synsets to descriptors...")
    syn_to_desc = get_synset_descriptors(propagated_canonical)
    with open(SYN_TO_DESC_FILE, "w") as f:
        json.dump(syn_to_desc, f, indent=4)
    print("Created and saved synset to descriptor file.")
    return syn_to_desc



if __name__ == '__main__':
    pass