"""BDDL -- Behavior Domain Definition Language.

BDDL defines a symbolic language for specifying household activities as
initial-state / goal-state condition pairs over a set of typed objects and
predicates.

The public API is through the :class:`~bddl.knowledge_base.KnowledgeBase`
and its models::

    from bddl.knowledge_base import KnowledgeBase, Task, Synset

    kb = KnowledgeBase(populate=True)
    task = kb.get_task("cleaning_up_after_a_meal-0")

    # Evaluate goal with a user-supplied callback
    def my_eval(predicate_cls, *entity_names):
        ...  # return True/False from your simulator

    success, results = task.check_goal(my_eval)

Predicate classes (used as callback arguments and dict keys) are in
:mod:`bddl.predicates`.
"""
