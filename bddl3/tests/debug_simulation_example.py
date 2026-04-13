from bddl.activity import (
    Conditions,
    evaluate_goal_conditions,
    get_goal_conditions,
    get_ground_goal_state_options,
    get_object_scope,
    get_reward,
)


def main():
    # Set parameters
    activity = "cleaning_up_after_a_meal"
    activity_definition = 0
    desired_simulator = "behavior-1k"

    # Load and compile activity
    print(f"Loading activity {activity}")
    conds = Conditions(activity, activity_definition, desired_simulator)
    scope = get_object_scope(conds)

    # Simple in-memory state tracker for trivial simulation
    state = {"unary": set(), "binary": set()}

    def set_state(literals):
        for literal in literals:
            is_predicate = not (literal[0] == "not")
            predicate, *objects = literal[1] if (literal[0] == "not") else literal
            if predicate == "inroom":
                continue
            key = (predicate, *objects)
            if is_predicate:
                if len(objects) == 1:
                    state["unary"].add(key)
                else:
                    state["binary"].add(key)
            else:
                state["unary"].discard(key)
                state["binary"].discard(key)

    def evaluate_fn(predicate_name, *entities):
        key = (predicate_name, *entities)
        return key in state["unary"] or key in state["binary"]

    # Compile goal conditions and ground goal solutions
    goal_conds = get_goal_conditions(conds, scope, generate_ground_options=True)
    ground_goal_state_options = get_ground_goal_state_options(conds, scope, goal_conds)

    # Set intermediate state steps
    print()
    print("Setting state")
    set_state(conds.parsed_initial_conditions)
    set_state([["not", ["covered", "chair.n.01_2", "stain.n.01_1"]]])
    set_state([["not", ["covered", "chair.n.01_1", "stain.n.01_1"]]])
    set_state([["not", ["covered", "bowl.n.01_1", "stain.n.01_1"]]])
    set_state([["not", ["covered", "bowl.n.01_2", "stain.n.01_1"]]])
    set_state([["not", ["covered", "plate.n.04_1", "stain.n.01_1"]]])
    set_state([["not", ["covered", "plate.n.04_2", "stain.n.01_1"]]])

    # Evaluate compiled goal conditions on current state
    print()
    print("Evaluating")
    is_successful, satisfied = evaluate_goal_conditions(goal_conds, evaluate_fn)
    reward = get_reward(ground_goal_state_options, evaluate_fn)
    print(is_successful)
    print(satisfied)
    print(reward)
    input()

    set_state([["not", ["covered", "plate.n.04_3", "stain.n.01_1"]]])
    set_state([["not", ["covered", "plate.n.04_4", "stain.n.01_1"]]])
    set_state([["not", ["covered", "cup.n.01_1", "stain.n.01_1"]]])
    set_state([["not", ["covered", "cup.n.01_2", "stain.n.01_1"]]])
    set_state([["inside", "hamburger.n.01_1", "sack.n.01_1"]])
    set_state([["inside", "hamburger.n.01_2", "sack.n.01_1"]])

    # Evaluate compiled goal conditions on current state
    print()
    print("Evaluating")
    is_successful, satisfied = evaluate_goal_conditions(goal_conds, evaluate_fn)
    reward = get_reward(ground_goal_state_options, evaluate_fn)
    print(is_successful)
    print(satisfied)
    print(reward)
    input()

    set_state([["ontop", "sack.n.01_1", "floor.n.01_1"]])
    set_state([["not", ["covered", "table.n.02_1", "stain.n.01_1"]]])

    # Evaluate compiled goal conditions on current state
    print()
    print("Evaluating")
    is_successful, satisfied = evaluate_goal_conditions(goal_conds, evaluate_fn)
    reward = get_reward(ground_goal_state_options, evaluate_fn)
    print(is_successful)
    print(satisfied)
    print(reward)
    input()

    return is_successful


is_successful = main()
print()
print("Final result:", is_successful)
