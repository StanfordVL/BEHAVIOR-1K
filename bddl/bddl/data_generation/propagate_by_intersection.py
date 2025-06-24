import pathlib
import json 
from nltk.corpus import wordnet as wn

PROPAGATED_CANONICAL = pathlib.Path(__file__).parents[1] / "generated_data" / "propagated_annots_canonical.json"


def get_intersection(synset, property_map, canonical):
    if "children" in synset: # if synset node has children
        props = set()
        for i in range(len(synset["children"])):
            child_synset = synset["children"][i]
            child_props = get_intersection(child_synset, property_map, canonical)
            if i == 0:
                props = set(child_props)
            else:
                props = props.intersection(child_props)
        property_map[synset["name"]] = props # add these once they have been intersected
        return props
    else: # base case
        if synset["name"] in canonical: # if the synsetified name shows up in canonical
            props = canonical[synset["name"]] # get its properties
            property_map[synset["name"]] = props
        else: # if the synsetified name isn't in canonical
            for elem in canonical: # check every synset of canonical to see if it's a synonym of variable `synset`
                try: # handle crashes from trying to iterate through custom synsets
                    x = wn.synset(elem)
                except:
                    pass
                else:
                    if wn.synset(elem) == wn.synset(synset["name"]): # if there's a synonym
                        props = canonical[elem] # append its properties
                        property_map[synset["name"]] = props
        return props
      

def process_output(prop_map):
    dict = {}
    for synset in sorted(prop_map):
        dict[synset] = {}
        for elem in sorted(prop_map[synset]):
            dict[synset][elem] = {}
    return dict


# API

def create_get_save_propagated_canonical(hierarchy, annots_canonical):
    prop_map = {}
    get_intersection(hierarchy, prop_map, annots_canonical)
    made_dict = process_output(prop_map)
    with open(PROPAGATED_CANONICAL, "w") as f:
      json.dump(made_dict, f, indent=2)
    return made_dict
    

if __name__ == '__main__':
    pass
