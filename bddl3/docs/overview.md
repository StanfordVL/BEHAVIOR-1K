# Overview

The BDDL codebase has two primary sections: activity definition and interface with the simulator. Given a simulator and an agent deployed in it, BDDL is typically used to instantiate a BEHAVIOR activity definition in the simulator, then check the agent's progress/success at every step. 

## Activity definition

### Summary

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

### BDDL language 

BDDL includes two types of files: the **domain** file and the **problem** file. There is one domain file per simulator, and one problem file per activity definition (in this sense, "problem" and "activity definition" are interchangeable). 

The domain contains three sections: domain name (`(domain <domain_name>)`); requirements (`(:requirements :strips :adl)` as BDDL relies on these); and predicates, a list of predicates with fields indicating their arity. See the example created for iGibson 2.0 [here](https://github.com/StanfordVL/bddl/blob/master/bddl/activity_definitions/domain_igibson.bddl).

The problem, i.e. an activity definition, is more complex. It consists of a problem name, a domain, objects, initial condition, and goal condition. See examples in subdirectories [here](https://github.com/StanfordVL/bddl/tree/master/bddl/activity_definitions). 

By convention, the problem (`problem`) section should take the form of `(problem <behavior_activity>_<activity_instance>)`, where `activity_instance` is some identifying number that distinguishes this definition of `behavior_activity` from other definitions of the same activity, since a BEHAVIOR activity (e.g. "packing lunches" or "cleaning bathtub") can be defined multiple times to make multiple versions.

The domain (`:domain`) section should take the form of `(:domain <domain_name>)`, where `domain_name` matches to the domain `define`d in some domain file. 

The objects (`:objects`) section should contain all object instances involved in the activity definition, categorized. For example, for an activity definition with three instances of some category `mycat`, `:objects` should include the following line: `mycat_1 mycat_2 mycat_3 - mycat`. BDDL requires that object instances be written as `<category>_<unique_id_number>` where `category` is a WordNet synset (see the next section for details on the role of WordNet in BDDL). `:objects` should list and categorize every object instance used in the definition. 

The initial condition (`:init`) should consist of a list of ground atomic formulae. This means that `:init` cannot contain logical operators such as `and`, `forall`, or `forpairs`, and all objects involved in it must be instances - concrete object instances like `mycat_1` - and not variables that may indicate multiple possible instances (e.g. just `mycat`, which could be any instance with category `mycat`). `:init` can contain certain types of negated atomic formulae (using the `not` logical operator) - specifically, when the atomic formula is **not** involved in location of objects. So, a BDDL activity definition **can** have `(not (cooked mycat_1))` but it **cannot** have `(not (ontop mycat_1 othercat_1))`. This is for the sampler - it is difficult to sample "anywhere-but" efficiently and ecologically. 

Finally, the goal condition (`:goal`) should consist of one logical expression, likely a conjunction of clauses. This expression can use any of the standard logical operators used in the [Planning Domain Definition Language (PDDL)](https://planning.wiki/ref/pddl/problem), namely `and`, `or`, `not`, `imply`, `forall`, and `exists`. It can also use our custom operators: `forn`, `forpairs`, and `fornpairs`. Note that 

### Annotations for activity definition


### Creating your own activity definition


## Interface with a simulator 

