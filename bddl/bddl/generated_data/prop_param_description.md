# Example `propagated_annots_params.json` entry
```
{
    "<synset>": {
        # keys in this dictionary indicate all the properties that *do* apply to <synset>.
        # In this example, we have given all possible properties and their parameters for the 
        #   sake of documentation, but in reality no object has every property. 
        "cookable": {                           # <synset> can be in cooked/not cooked states
            "cook_temperature": <float>,        # temperature that the object has to have been at or over at some point for cooking to trigger. If <synset> has property nonSubstance, this means cooked(<object>) becomes True. If <synset> has property substance, this means the quantity that reached cook_temperature disappears and is replaced by substance_cooking_derivative_synset.
            "substance_cooking_derivative_synset": <str>    # if <synset> has property substance, the new synset created when <synset> reaches cook_temperature.
        },
        "cloth": {},                            # <synset> is a single-particle-depth deformable (height and width can be >1),
        "coldSource": {                         # <synset> is a source of cold that can change the temperature of other objects (essentially the same as a heatSource, but typically with temperature lower than room temp)
            "requires_toggled_on": <float>,     # whether toggled_on(<synset>) must be true for it to heat other objects
            "requires_closed": <float>,         # whether closed(<synset>) must be true for it to heat an object inside it
            "requires_inside": <float>,         # whether inside(<other object>, <synset>) must be true for <synset> to cool <other object>,
            "temperature": <float>,             # temperature that <synset> emits when all requirements are fulfilled
            "heating_rate": <float>             # rate at which surroundings have their temperature changed             
        },
        "deformable": {},                       # <synset> is a deformable (a softBody, cloth, or rope)
        "diceable": {                           # if a slicer is applied to <synset>, it will disappear and be replaced by uncooked_diceable_derivative_synset
            "uncooked_diceable_derivative_synset": <str>   # synset that will be created if slicer is applied to <synset>
        },
        "drapeable": {},                        # <synset> can be in a draped(<synset>, <other object>) statement
        "flammable": {                          # <synset> can be in on_fire/not on_fire states
            "ignition_temperature": <float>,    # temperature that the object has to be at or over *due to exposure to a toggled_on(<fireSource>) or another object for which on_fire is True* for on_fire(<synset>) to become True
            "fire_temperature": <float>,        # temperature that on_fire(<synset>) emits, much like a heatSource
            "heating_rate": <float>,            # rate at which surroundings have their temperature changed when on_fire(<synset>) is True
            "distance_threshold": <float>:      # how far out radially the temperature effects reach 
        },
        "freezable": {},                        # <synset> can be in frozen/not frozen states
        "heatSource": {                         # <synset> is a source of heat that can change the temperature of other objects under certain conditions
            "requires_toggled_on": <float>,     # whether toggled_on(<synset>) must be true for it to heat other objects
            "requires_closed": <float>,         # whether closed(<synset>) must be true for it to heat an object inside it
            "requires_inside": <float>,         # whether inside(<other object>, <synset>) must be true for <synset> to heat <other object>,
            "temperature": <float>,             # temperature that <synset> emits when all requirements are fulfilled
            "heating_rate": <float>             # rate at which surroundings have their temperature changed 
        },                       
        "heatable": {                           # <synset> can be in hot/not hot states
            "heat_temperature": <float>         # temperature that the object has to be at or over for hot(<synset>) to be True
        }, 
        "liquid": {},                           # <synset> is a liquid
        "microPhysicalSubstance": {},           # <synset> is a microPhysicalSubstance (a physicalSubstance with minute particles, like flour)
        "macroPhysicalSubstance": {},           # <synset> is a macroPhysicalSubstance (a physicalSubstance with larger particles, like a blueberry)
        "meltable": {
            "meltable_derivative_synset": <str> # synset that will be created if <synset> is heated past its melting point at any time. <synset> will disappear, i.e. real(<synset>) will become False, and real(<meltable_derivative_synset>) will become True. 
        },
        "nonSubstance": {},                     # <synset> is not a substance
        "particleApplier": {                    # <synset> can release a substance if insource(<particleApplier>, <substance>) is True and given conditions are met
            "method": <int>,                    # method with which particles are release and apply onto a nonSubstance - 1 means projection, like a spray from a spray bottle
            "conditions": {                     # mapping of substances that this particleApplier can apply
                <substance>: [                  # substance that <synset> can apply if the listed conditions are true
                    [                           # state-boolean value pair such that state-indicated-by-<int>(<synset>) must have value <bool> for <substance> to be applied
                        <int>,                  
                        <bool>
                    ]
                ]
            }
        },
        "particleRemover": {                    # <synset> can cause substances to be removed from another object. Furthermore, it can be in a saturated(<synset>, substance) statement with some substance. 
            "conditions": {                     # mapping of substances that <synset> can remove off another object to the conditions that need to be met for that substance to be removed. The conditions may be an empty list, meaning the removal can be done with just <synset>, or they may be nontrivial, often because another substance needs to be saturated into <synset> for the removal to happen. For example, cleaning mud.n.03 off an object with a rag.n.01 requires that the rag.n.01 be saturated with water.n.06.
                <removable_substance>: [        # a substance that <synset> can remove
                    [
                        <int>,                  # state that needs to apply to <synset>
                        <removing_substance>    # a substance that <synset> needs to be saturated with to remove <removable_substance>
                    ]
                ],
                ...
            }
        },
        "particleSink": {                       # <synset> causes substances to disappear from the scene
            "conditions": {
                <removable_substance>: [        # same format as particleRemover conditions entries
                    <int>,
                    <bool>
                ]
            },
            "default_physical_conditions": [    # any requirements to remove physicalSubstances not specified in conditions
                [                               # same format as particleRemover conditions entries. If it's null/None, then physicalSubstances can't be removed by <synset>.
                    <int>,
                    <bool>
                ]
            ], 
            "default_visual_conditions": [      # any requirements to remove visualSubstances not specified in conditions.
                [                               # same format as particleRemover conditions entries. If it's null/None, then physicalSubstances can't be removed by <synset>.
                    <int>,                          
                    <bool>
                ]
            ]
        },
        "particleSource": {                     # <synset> can generate and release a substance on its own if given conditions are met
            "conditions": {
                <substance>: [                  # a substance that <synset> can generate and apply
                    [
                        <int>,                  
                        <bool>
                    ]
                ]
            }
        },
        "physicalSubstance": {},                # <synset> is a physicalSubstance (a substance with mass and finite friction)
        "rigidBody": {},                        # <synset> is a rigidBody, i.e. a non-deformable nonSubstance
        "rope": {},                             # <synset> is a single-particle-width, single-particle-depth deformable (length can be >1)
        "sliceable": {                          # if a slicer is applied to <synset>, it will disappear and be replaced by two sliceable_derivative_synset instances
            "sliceable_derivative_synset": <str>    # synset of which two will be created if slicer is applied to <synset>
        },
        "slicer": {},                           # <synset> can slice a sliceable or dice a diceable
        "softBody": {},                         # <synset> is a deformable where every dimension can be >1
        "substance": {},                        # <synset> is a substance - one or more of physicalSubstance, visualSubstance, or liquid
        "toggleable": {},                       # <synset> can be in toggled_on/not toggled_on states
    }
}
```