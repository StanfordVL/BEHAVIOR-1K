"""BDDL -- Behavior Domain Definition Language.

BDDL defines a symbolic language for specifying household activities as
initial-state / goal-state condition pairs over a set of typed objects and
predicates.  It extends a subset of PDDL with quantifiers (``forn``,
``forpairs``, ``fornpairs``) suited to household task descriptions.

Quick start::

    from bddl.task import Task

    task = Task("cleaning_up_after_a_meal", 0)

    # Evaluate goal with a user-supplied callback
    def my_eval(predicate_name, *entity_names):
        ...  # return True/False from your simulator

    success, results = task.check_goal(my_eval)

See :mod:`bddl.task` for the high-level interface, :mod:`bddl.activity` for
lower-level condition compilation, and :mod:`bddl.condition_evaluation` for
the expression tree implementation.
"""
