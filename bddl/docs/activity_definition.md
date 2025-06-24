# Activity definition

## Summary

BDDL activities are defined by a set of **objects** in a scene, a ground **initial condition** that the scene configuration satisfies when the agent starts the activity, and a **goal condition** logical expression that the scene configuration must satisfy for the agent to reach success. The following example demonstrates this:

```
(define 
    (problem cleaning_the_pool_simplified)
    (:domain igibson)

    (:objects
     	pool.n.01_1 - pool.n.01
    	floor.n.01_1 - floor.n.01
    	scrub_brush.n.01_1 - scrub_brush.n.01
        sink.n.01_1 - sink.n.01
    	agent.n.01_1 - agent.n.01
    )
    
    (:init 
        (ontop pool.n.01_1 floor.n.01_1) 
        (stained pool.n.01_1) 
        (ontop scrub_brush.n.01_1 floor.n.01_1) 
        (inroom floor.n.01_1 garage) 
        (inroom sink.n.01_1 storage_room)
        (ontop agent.n.01_1 floor.n.01_1)
    )
    
    (:goal 
        (and 
            (ontop ?pool.n.01_1 ?floor.n.01_1) 
            (not 
                (stained ?pool.n.01_1)
            ) 
            (ontop ?scrub_brush.n.01_1 ?shelf.n.01_1) 
        )
    )
)
```
The `:objects` and `:init` sections specify the initial state as a set of objects and a set of initial atomic formulae the objects must satisfy at the start of the task. The `:goal` section specifies the expression that the objects must satisfy for the task to be considered successfully completed, i.e. what the agent needs to achieve.

## BDDL language 

BDDL includes two types of files: the **domain** file and the **problem** file. There is one domain file per simulator, and one problem file per activity definition. We use "problem" only to keep consistent with the Planning Domain Definition Language (PDDL), and from hereon will use only "activity definition". In general, BDDL activity definitions are logical expressions using standard logical operators, some custom operators, objects instances as ground terms, object categories as variables, and object properties as predicates. 

### Domain file 

The domain contains three sections: domain name (`(domain <domain_name>)`); requirements (`(:requirements :strips :adl)` as BDDL relies on these); and predicates, a list of predicates with fields indicating their arity. See the example created for iGibson 2.0 [here](https://github.com/StanfordVL/bddl/blob/master/bddl/activity_definitions/domain_igibson.bddl).

### Activity definition file header

The pactivity definition is more complex. It consists of an activity definition name, a domain, objects, initial condition, and goal condition. See examples in subdirectories [here](https://github.com/StanfordVL/bddl/tree/master/bddl/activity_definitions). 

By convention, the problem (`problem`) section should take the form of `(problem <behavior_activity>_<activity_instance>)`, where `activity_instance` is some identifying number that distinguishes this definition of `behavior_activity` from other definitions of the same activity, since a BEHAVIOR activity (e.g. "packing lunches" or "cleaning bathtub") can be defined multiple times to make multiple versions.

### Activity definition file domain 

The domain (`:domain`) section should take the form of `(:domain <domain_name>)`, where `domain_name` matches to the domain `define`d in some domain file. 

### Activity definition file `:objects` section

The objects (`:objects`) section should contain all object instances involved in the activity definition, categorized. For example, for an activity definition with three instances of some category `mycat`, `:objects` should include the following line: `mycat_1 mycat_2 mycat_3 - mycat`. BDDL requires that object instances be written as `<category>_<unique_id_number>` where `category` is a WordNet synset (see the **Annotations for activity definition** section for details on the role of WordNet in BDDL). `:objects` should list and categorize every object instance used in the definition. 

### Activity definition file `:init` section

The initial condition (`:init`) should consist of a list of ground atomic formulae. This means that `:init` cannot contain logical operators such as `and`, `forall`, or `forpairs`, and all objects involved in it must be instances - concrete object instances like `mycat_1` - and not variables that may indicate multiple possible instances (e.g. just `mycat`, which could be any instance with category `mycat`). `:init` can contain certain types of negated atomic formulae (using the `not` logical operator) - specifically, when the atomic formula is **not** involved in location of objects. So, a BDDL activity definition **can** have `(not (cooked mycat_1))` but it **cannot** have `(not (ontop mycat_1 othercat_1))`. This is for the sampler - it is difficult to sample "anywhere-but" efficiently and ecologically. 

The `:init` section has some additional requirements. For this, we note that all objects in BDDL are either **scene objects** or **additional objects**. **Scene objects** are those that are already available by default in the simulator scene that you are instantiating an activity definition in. **Additional objects** are those that will be added to the scene by the sampler. To make sure that the `:init` that a sampler needs to handle is not underspecfied, BDDL requires the following: 
- Every scene object involved in the activity definition must appear in exactly one `inroom` atomic formula specifying the room it's in, e.g. `(inroom myscenecat_1 kitchen)`. Note that you do not need to list all of the scene's default objects in your activity definition, you only need to list the ones that are relevant to you.
- Every additional object must appear in a binary atomic formula that specifies its position relative to a scene object either directly or transitively. So, the following is acceptable, because it has a direct positioning:
```
(ontop myaddcat_1 myscenecat_1)
```
And this is also acceptable, because it has indirect positioning for some additional objects but they are all ultimately positioned relative to a scene object:
```
(ontop myaddcat_1 myaddcat_2)
(ontop myaddcat_2 myscenecat_1)
(ontop myaddcat_3 myaddcat_1)
```
The following is not acceptable because even though all additional objects appear as arguments to a positional predicate, they are not all placed relative to scene objects:
```
(ontop myaddcat_1 myscenecat_1)
(ontop myaddcat_2 myaddcat_3)
```

### Activity definition file `:goal` section

Finally, the goal condition (`:goal`) should consist of one logical expression, likely a conjunction of clauses. This expression can use any of the standard logical operators used in the [Planning Domain Definition Language (PDDL)](https://planning.wiki/ref/pddl/problem), namely `and`, `or`, `not`, `imply`, `forall`, and `exists`. It can also use our custom operators: `forn`, `forpairs`, and `fornpairs`. These custom operators are defined as follows:
- `forn`: for some non-negative integer `n` and some object category `mycat`, the child condition must hold true for at least `n` instances of category `mycat`
- `forpairs`: for two object categories `mycat` and `othercat`, the child conditiono must hold true for some one-to-one mapping of object instances of `mycat` to object instances of `othercat` that covers all instances of at least one of the two categories
- `fornpairs`: for some non-negative integer `n` and two object categories `mycat` and `othercat`, the child condition must hold true for at least `n` pairs of instances of `mycat` and instances of `othercat` that follow a one-to-one mapping. 

Unlike the `:init` section, where object instances can be used as terms but object categories cannot, `:goal` can use object categories as *bound variables*. A variable must be bound in a quantifier to be concrete and not ambiguous, and in `:goal` we have several quantifiers available: `forall`, `exists`, `forn`, `forpairs`, and `fornpairs`. In the `:goal`, BDDL allows any of these quantifiers to be used to bind categories, to make goal conditions that are more flexible and concise than those that specify object instances everywhere. 

## Annotations for activity definition

Using BDDL to make activity definition content requires two types of annotations: 
1. Annotations of object categories as [WordNet](https://wordnet.princeton.edu/) *synsets* (terms for distinct concepts) that follow the WordNet hierarchy. 
2. Annotations mapping object categories to properties they exhibit. 

### Hierarchical object category annotations 
The object categories in BDDL are all WordNet synsets. WordNet synsets are single terms for groups of cognitive synonyms, removing the ambiguity coming from multiple words meaning the same thing or one word having multiple senses. In BDDL, each category means exactly what its synset refers to. 

BDDL object categories (such as `mycat` and `othercat` above) have the following syntax: `<word/phrase>.n.<ID number>`. This is a subset of WordNet's synset syntax - the `n` in the middle of the BDDL category syntax indicates "noun". In WordNet, there are a few other options for other parts of speech that are not relevant to BDDL. 

The list of BDDL object categories is found in [`objectmodeling.csv`](https://github.com/StanfordVL/bddl/blob/master/utils/objectmodeling.csv). Note that these are specifically categories that also have at least one 3D model in the BEHAVIOR Object Dataset.

The synset formulation allows BDDL objects to follow the WordNet hierarchy, meaning that an object will not only be seen as an instance of its own category, but also all parent categories. 

The hierarchy is specified in `hierarchy*.json` files. These are generated by running [`hierarchy_generator.py`](https://github.com/StanfordVL/bddl/blob/master/utils/hierarchy_generator.py); see the script for more details on the format of the various hierarchies generated. 

### Object-to-property mapping
As stated above, objects are terms in BDDL and their properties are predicates. BDDL currently requires support for the set of properties specified in [`domain_igibson.bddl`](https://github.com/StanfordVL/bddl/blob/master/bddl/activity_definitions/domain_igibson.bddl) - all the unary predicates are states of individual objects.

Of course, not every property applies to every object. Therefore, BDDL requires a mapping of every object to the properties it has. For BDDL's original supported objects, these mappings come from crowdsourced annotations. These annotations are found in [`synsets_to_filtered_properties.json`](https://github.com/StanfordVL/bddl/blob/master/utils/synsets_to_filtered_properties.json). Furthermore, simply mapping is not always enough - some properties require additional parameters. For example, different items may all be cookable and burnable, but cook and burn at different temperatures, so they not only need to be annotated as `cookable` or `burnable`, but also with a `cook_temperature` and `burn_temperature`. `synsets_to_filtered_properties.json` shows the correct syntax for the annotations. 

Finally, these properties are filtered to be inherited through the hierarchy. More concretely, the hierarchy generation script enforces that in `hierarchy*.json`, the properties of any non-leaf category will be the intersection of the properties of all its descendant categories; only the leaf categories' properties are determined by their crowdsourced annotation. Note that `synsets_to_filtered_properties.json` does **not** obey this rule, so you cannot use it directly to determine properties, only to enter and store original annotations. You must use `hierarchy*.json` for determining properties that actually apply in the activity definition. 

### Adding your own object to BDDL 

Adding your own object to BDDL requires providing the above annotations. Do not do so unless you support an appropriate 3D object model in your simulator for the category you are adding. For instructions on how to add an object model to iGibson 2.0, click [here]().

Once you have a usable object model, do the following: 
1. Find the most suitable synset for your object category in WordNet. 
2. Add your object's synset, as well as a non-synset label if you want, to `bddl/utils/objectmodeling.csv`. **Note:** if the synset you chose is already in `objectmodeling.csv`, you can stop here!
3. Decide which of the properties in `bddl/bddl/activity_definitions/domain_igibson.bddl` apply to your object and annotate them in `bddl/utils/synsets_to_filtered_properties.json`, using the correct syntax and providing any necessary paramaters. 
4. *Only if your simulator definitely supports each property for this object*, edit `bddl/utils/prune_object_property.py` to include your object and all its supported properties, and specify an outfile for your purposes; if you are using iGibson, keep the default value `synsets_to_filtered_properties_pruned_igibson.json`. 
5. Run `bddl/utils/hierarchy_generator.py` to generate the various hierarchy JSONs, wihch will now contain your synset associated with its properties. 

## Creating your own activity definition
Use the following steps to create your own BDDL activity. 
1. **Choose an activity you want to define,** like "putting away groceries" or "washing dishes". You can choose from the [list of BEHAVIOR activities](https://behavior.stanford.edu/activity-annotation) or you can make your own. 
2. **Choose a method of making your definition.** There are multiple ways to do so: 
    a. Using our visual-BDDL annotator [here](https://behavior.stanford.edu/activity-annotation). This requires no setup and offers helpful constraints, but your definition is not guaranteed to fit in a scene even if it is syntactically correct. 
    b. Using a local version of our annotation system (download the [code](https://github.com/StanfordVL/behavior-activity-annotator)). This requires more setup, offers helpful constraints, and if you attach a simulator to test sampling (example implementation in iGibson 2.0 [here](TODO)), you can edit your definition until it fits. 
    c. Writing your own BDDL file directly. This requires no setup, and has neither helpful constraints nor limiting constraints. Your definition is not guaranteed to be syntactically correct or fit in a scene.
3. **Define the activity in BDDL.** 
    a/b. If you are using our annotation interface, whether locally or online, you will find instructions at the [landing page](https://behavior.stanford.edu/activity-annotation) and more detailed instructions in the interface itself. 
    c. If you are writing BDDL directly, refer to the earlier section of this page as well as [existing definitions](https://github.com/StanfordVL/bddl/tree/master/bddl/activity_definitions) for guidance. 