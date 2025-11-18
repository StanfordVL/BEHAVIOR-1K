from bddl.activity import *
from bddl.trivial_backend import TrivialBackend, TrivialGenericObject, TrivialSimulator


def main():
    # Set parameters
    activity = "cleaning_up_after_a_meal"
    activity_definition = 0
    desired_simulator = "omnigibson"       # This tells us which domain file we will get predicates from, but we can still plug a different simulator into the actual scope. 
    
    # Load and compile activity, stub backend, and stub simulator
    print(f"Loading activity {activity}")
    conds = Conditions(activity, activity_definition, desired_simulator)
    backend = TrivialBackend()
    simulator = TrivialSimulator()
    scope = get_object_scope(conds)
    for name in scope:
        scope[name] = TrivialGenericObject(name, simulator)

    # Compile goal conditions and (optionally) ground goal solutions (in parsed format)
    goal_conds = get_goal_conditions(conds, backend, scope, generate_ground_options=True)
    ground_goal_state_options = get_ground_goal_state_options(conds, backend, scope, goal_conds)
    
    # Set intermediate state steps
    print(); print("Setting state")
    simulator.set_state(conds.parsed_initial_conditions)
    simulator.set_state([["not", ["covered", "chair.n.01_2", "stain.n.01_1"]]])
    simulator.set_state([["not", ["covered", "chair.n.01_1", "stain.n.01_1"]]])
    simulator.set_state([["not", ["covered", "bowl.n.01_1", "stain.n.01_1"]]])
    simulator.set_state([["not", ["covered", "bowl.n.01_2", "stain.n.01_1"]]])
    simulator.set_state([["not", ["covered", "plate.n.04_1", "stain.n.01_1"]]])
    simulator.set_state([["not", ["covered", "plate.n.04_2", "stain.n.01_1"]]])
    
    # Evaluate compiled goal conditions on current state
    print(); print("Evaluating")
    is_successful, satisfied = evaluate_goal_conditions(goal_conds)
    # reward = max(len(evaluate_goal_conditions(option)[-1]["satisfied"])/float(len(option)) for option in ground_goal_state_options)
    reward = get_reward(ground_goal_state_options)
    print(is_successful); print(satisfied); print(reward); input()
    
    simulator.set_state([["not", ["covered", "plate.n.04_3", "stain.n.01_1"]]])
    simulator.set_state([["not", ["covered", "plate.n.04_4", "stain.n.01_1"]]])
    simulator.set_state([["not", ["covered", "cup.n.01_1", "stain.n.01_1"]]])
    simulator.set_state([["not", ["covered", "cup.n.01_2", "stain.n.01_1"]]])
    simulator.set_state([["inside", "hamburger.n.01_1", "sack.n.01_1"]])
    simulator.set_state([["inside", "hamburger.n.01_2", "sack.n.01_1"]])
    
    # Evaluate compiled goal conditions on current state
    print(); print("Evaluating")
    is_successful, satisfied = evaluate_goal_conditions(goal_conds)
    reward = get_reward(ground_goal_state_options)
    print(is_successful); print(satisfied); print(reward); input()
    
    simulator.set_state([["ontop", "sack.n.01_1", "floor.n.01_1"]])
    simulator.set_state([["not", ["covered", "table.n.02_1", "stain.n.01_1"]]])
    
    # Evaluate compiled goal conditions on current state
    print(); print("Evaluating")
    is_successful, satisfied = evaluate_goal_conditions(goal_conds)
    reward = get_reward(ground_goal_state_options)
    print(is_successful); print(satisfied); print(reward); input()

    return is_successful


is_successful = main()
print()
print("Final result:", is_successful)
