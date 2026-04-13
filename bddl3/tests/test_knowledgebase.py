"""Tests for the BDDL KnowledgeBase, condition evaluation, and predicate system."""

import pytest

from bddl.knowledge_base import KnowledgeBase, Task, Synset
from bddl.condition_evaluation import compile_state, evaluate_state
from bddl.logic_base import Expression
from bddl.predicates import (
    Predicate,
    UnaryPredicate,
    BinaryPredicate,
    OnTop,
    Inside,
    Cooked,
    Covered,
    Future,
    Real,
    TOKEN_TO_PREDICATE,
)
from bddl.transition_rules import CookingRecipe, MachineRecipe


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def kb():
    """Shared KB instance without WordNet (fast)."""
    return KnowledgeBase(verbose=False)


@pytest.fixture(scope="session")
def kb_wn():
    """Shared KB instance with WordNet (slow, for definition tests)."""
    return KnowledgeBase(verbose=False, load_wordnet=True)


# ---------------------------------------------------------------------------
# KB Population
# ---------------------------------------------------------------------------


class TestKBPopulation:
    def test_kb_loads_without_wordnet(self, kb):
        assert len(kb.all_synsets()) > 3000
        assert len(kb.all_tasks()) > 1000
        assert len(kb.all_categories()) > 0
        assert len(kb.all_transition_rules()) > 0

    def test_kb_loads_with_wordnet(self, kb_wn):
        assert len(kb_wn.all_synsets()) > 3000
        assert len(kb_wn.all_tasks()) > 1000

    def test_synset_definition_empty_without_wordnet(self, kb):
        s = kb.get_synset("table.n.02")
        assert s.definition == ""

    def test_synset_definition_present_with_wordnet(self, kb_wn):
        s = kb_wn.get_synset("table.n.02")
        assert len(s.definition) > 0
        assert "furniture" in s.definition.lower() or "flat" in s.definition.lower()

    def test_synset_is_custom_with_wordnet(self, kb_wn):
        # "table.n.02" is a real WordNet synset
        s = kb_wn.get_synset("table.n.02")
        assert s.is_custom is False
        # Custom synsets (derivative objects like "cooked__X") should be marked custom
        custom_synsets = [s for s in kb_wn.all_synsets() if s.is_custom]
        assert len(custom_synsets) > 0


# ---------------------------------------------------------------------------
# Synset Model
# ---------------------------------------------------------------------------


class TestSynset:
    def test_synset_hierarchy(self, kb):
        s = kb.get_synset("table.n.02")
        assert len(s.parents) > 0
        assert any(p.name == "furniture.n.01" for p in s.parents)
        assert len(s.children) > 0
        assert len(s.ancestors) > len(s.parents)
        assert len(s.descendants) >= len(s.children)

    def test_synset_is_leaf(self, kb):
        # Find a leaf synset (one with categories)
        leaf = None
        for s in kb.all_synsets():
            if len(s.categories) > 0:
                leaf = s
                break
        assert leaf is not None
        assert leaf.is_leaf is True
        # Non-leaf
        s = kb.get_synset("furniture.n.01")
        assert s.is_leaf is False

    def test_synset_abilities(self, kb):
        # knife.n.01 should have "slicer" ability
        s = kb.get_synset("knife.n.01")
        abilities = s.abilities
        assert isinstance(abilities, dict)
        assert "slicer" in abilities

    def test_synset_is_substance(self, kb):
        s = kb.get_synset("water.n.06")
        assert s.is_substance is True
        s2 = kb.get_synset("table.n.02")
        assert s2.is_substance is False

    def test_synset_categories(self, kb):
        # breakfast_table is a category under table.n.02's subtree
        cat = kb.get_category("breakfast_table")
        assert cat is not None
        assert cat.synset is not None

    def test_synset_required_meta_links(self, kb):
        # An openable synset should require "joint" meta link
        s = kb.get_synset("microwave.n.02")
        if "openable" in s.abilities:
            assert "joint" in s.required_meta_links


# ---------------------------------------------------------------------------
# Task Model
# ---------------------------------------------------------------------------


class TestTask:
    def test_task_exists(self, kb):
        task = kb.get_task("cleaning_up_after_a_meal-0")
        assert task is not None
        assert task.name == "cleaning_up_after_a_meal-0"

    def test_compile_returns_compiled_task(self, kb):
        from bddl.knowledge_base.models import CompiledTask

        task = kb.get_task("cleaning_up_after_a_meal-0")
        ct = task.compile(scene_layout={})
        assert isinstance(ct, CompiledTask)
        assert ct.task is task

    def test_task_check_goal_all_false(self, kb):
        ct = kb.get_task("cleaning_up_after_a_meal-0").compile(scene_layout={})
        ok, results = ct.check_goal(lambda cls, *e: False)
        assert ok is False
        assert len(results["unsatisfied"]) > 0

    def test_task_check_goal_tracks_satisfied(self, kb):
        ct = kb.get_task("cleaning_up_after_a_meal-0").compile(scene_layout={})
        ok, results = ct.check_goal(lambda cls, *e: True)
        assert len(results["satisfied"]) + len(results["unsatisfied"]) == len(ct.goal_conditions)

    def test_task_object_scope(self, kb):
        ct = kb.get_task("cleaning_up_after_a_meal-0").compile(scene_layout={})
        assert isinstance(ct.object_scope, set)
        assert len(ct.object_scope) > 0
        for name in ct.object_scope:
            assert isinstance(name, str)

    def test_task_parsed_objects(self, kb):
        ct = kb.get_task("cleaning_up_after_a_meal-0").compile(scene_layout={})
        assert isinstance(ct.parsed_objects, dict)
        for cat, instances in ct.parsed_objects.items():
            assert isinstance(cat, str)
            assert isinstance(instances, list)
            assert all(isinstance(i, str) for i in instances)

    def test_task_ground_goal_state_options(self, kb):
        ct = kb.get_task("cleaning_up_after_a_meal-0").compile(scene_layout={})
        assert isinstance(ct.ground_goal_state_options, list)
        assert len(ct.ground_goal_state_options) > 0

    def test_concurrent_compilations(self, kb):
        """Two CompiledTasks from the same Task are independent."""
        task = kb.get_task("carrying_in_groceries-0")

        # Compile with different scene layouts
        ct1 = task.compile(scene_layout={
            "garage": {"car": 1, "floor": 1},
            "kitchen": {"fridge": 2, "floor": 1},
        })
        ct2 = task.compile(scene_layout={
            "garage": {"car": 1, "floor": 1},
            "kitchen": {"fridge": 4, "floor": 1},
        })

        # They should have different object scopes (different fridge counts)
        fridges_1 = {n for n in ct1.object_scope if "electric_refrigerator" in n}
        fridges_2 = {n for n in ct2.object_scope if "electric_refrigerator" in n}
        assert len(fridges_1) == 2
        assert len(fridges_2) == 4

        # Both should be independently evaluable
        ok1, _ = ct1.check_goal(lambda cls, *e: False)
        ok2, _ = ct2.check_goal(lambda cls, *e: False)
        assert isinstance(ok1, bool)
        assert isinstance(ok2, bool)

        # Task definition is unchanged
        assert "*" in task.definition


# ---------------------------------------------------------------------------
# Condition Evaluation with Stubs
# ---------------------------------------------------------------------------


class TestConditionEvaluation:
    def test_evaluate_predicate_callback_receives_class(self):
        scope = {"bowl.n.01_1", "table.n.02_1"}
        object_map = {"bowl.n.01": ["bowl.n.01_1"], "table.n.02": ["table.n.02_1"]}
        parsed = [["ontop", "bowl.n.01_1", "table.n.02_1"]]
        compiled = compile_state(parsed, scope=scope, object_map=object_map)

        received = []

        def eval_fn(pred_cls, *entities):
            received.append((pred_cls, entities))
            return True

        evaluate_state(compiled, eval_fn)
        assert len(received) == 1
        assert received[0][0] is OnTop
        assert received[0][1] == ("bowl.n.01_1", "table.n.02_1")

    def test_evaluate_conjunction(self):
        scope = {"bowl.n.01_1", "table.n.02_1", "stain.n.01_1"}
        object_map = {
            "bowl.n.01": ["bowl.n.01_1"],
            "table.n.02": ["table.n.02_1"],
            "stain.n.01": ["stain.n.01_1"],
        }
        # (and (ontop bowl table) (covered bowl stain))
        parsed = [["and", ["ontop", "bowl.n.01_1", "table.n.02_1"], ["covered", "bowl.n.01_1", "stain.n.01_1"]]]
        compiled = compile_state(parsed, scope=scope, object_map=object_map)

        # Both true -> True
        ok, _ = evaluate_state(compiled, lambda cls, *e: True)
        assert ok is True

        # One false -> False
        ok, _ = evaluate_state(compiled, lambda cls, *e: cls is OnTop)
        assert ok is False  # Covered returns False

    def test_evaluate_negation(self):
        scope = {"bowl.n.01_1", "stain.n.01_1"}
        object_map = {"bowl.n.01": ["bowl.n.01_1"], "stain.n.01": ["stain.n.01_1"]}
        # (not (covered bowl stain))
        parsed = [["not", ["covered", "bowl.n.01_1", "stain.n.01_1"]]]
        compiled = compile_state(parsed, scope=scope, object_map=object_map)

        # Covered is False -> NOT False = True
        ok, _ = evaluate_state(compiled, lambda cls, *e: False)
        assert ok is True

        # Covered is True -> NOT True = False
        ok, _ = evaluate_state(compiled, lambda cls, *e: True)
        assert ok is False

    def test_evaluate_forall(self):
        scope = {
            "bowl.n.01_1", "bowl.n.01_2", "stain.n.01_1",
        }
        object_map = {
            "bowl.n.01": ["bowl.n.01_1", "bowl.n.01_2"],
            "stain.n.01": ["stain.n.01_1"],
        }
        # (forall (?x - bowl.n.01) (not (covered ?x stain.n.01_1)))
        parsed = [
            ["forall", ["?x", "-", "bowl.n.01"], ["not", ["covered", "?x", "stain.n.01_1"]]]
        ]
        compiled = compile_state(parsed, scope=scope, object_map=object_map)

        # All not covered -> True
        ok, _ = evaluate_state(compiled, lambda cls, *e: False)
        assert ok is True

        # All covered -> False (negation fails)
        ok, _ = evaluate_state(compiled, lambda cls, *e: True)
        assert ok is False

    def test_evaluate_exists(self):
        scope = {
            "bowl.n.01_1", "bowl.n.01_2", "table.n.02_1",
        }
        object_map = {
            "bowl.n.01": ["bowl.n.01_1", "bowl.n.01_2"],
            "table.n.02": ["table.n.02_1"],
        }
        # (exists (?x - bowl.n.01) (ontop ?x table.n.02_1))
        parsed = [
            ["exists", ["?x", "-", "bowl.n.01"], ["ontop", "?x", "table.n.02_1"]]
        ]
        compiled = compile_state(parsed, scope=scope, object_map=object_map)

        # Only bowl_1 is on table -> exists satisfied
        def eval_fn(cls, *entities):
            return cls is OnTop and entities[0] == "bowl.n.01_1"

        ok, _ = evaluate_state(compiled, eval_fn)
        assert ok is True

        # Neither on table -> exists fails
        ok, _ = evaluate_state(compiled, lambda cls, *e: False)
        assert ok is False


    def test_evaluate_disjunction(self):
        scope = {
            "bowl.n.01_1", "table.n.02_1", "floor.n.01_1",
        }
        object_map = {
            "bowl.n.01": ["bowl.n.01_1"],
            "table.n.02": ["table.n.02_1"],
            "floor.n.01": ["floor.n.01_1"],
        }
        # (or (ontop bowl table) (ontop bowl floor))
        parsed = [
            ["or", ["ontop", "bowl.n.01_1", "table.n.02_1"], ["ontop", "bowl.n.01_1", "floor.n.01_1"]]
        ]
        compiled = compile_state(parsed, scope=scope, object_map=object_map)

        # One branch true -> OR satisfied
        def on_table(cls, *entities):
            return cls is OnTop and entities[1] == "table.n.02_1"

        ok, _ = evaluate_state(compiled, on_table)
        assert ok is True

        # Neither branch true -> OR fails
        ok, _ = evaluate_state(compiled, lambda cls, *e: False)
        assert ok is False

    def test_evaluate_implication(self):
        scope = {"bowl.n.01_1", "table.n.02_1", "stain.n.01_1"}
        object_map = {
            "bowl.n.01": ["bowl.n.01_1"],
            "table.n.02": ["table.n.02_1"],
            "stain.n.01": ["stain.n.01_1"],
        }
        # (imply (ontop bowl table) (not (covered bowl stain)))
        # "if bowl is on table, then bowl must not be covered in stain"
        parsed = [
            ["imply", ["ontop", "bowl.n.01_1", "table.n.02_1"], ["not", ["covered", "bowl.n.01_1", "stain.n.01_1"]]]
        ]
        compiled = compile_state(parsed, scope=scope, object_map=object_map)

        # Antecedent false -> implication true regardless of consequent
        ok, _ = evaluate_state(compiled, lambda cls, *e: False)
        assert ok is True

        # Antecedent true, consequent true (covered=True, NOT covered=False) -> fails
        ok, _ = evaluate_state(compiled, lambda cls, *e: True)
        assert ok is False

        # Antecedent true, consequent true (ontop=True, covered=False -> NOT covered=True)
        def ontop_not_covered(cls, *entities):
            return cls is OnTop

        ok, _ = evaluate_state(compiled, ontop_not_covered)
        assert ok is True

    def test_evaluate_nested_forall_negation(self):
        """Test forall with negation: all bowls must NOT be covered."""
        scope = {
            "bowl.n.01_1", "bowl.n.01_2", "bowl.n.01_3", "stain.n.01_1",
        }
        object_map = {
            "bowl.n.01": ["bowl.n.01_1", "bowl.n.01_2", "bowl.n.01_3"],
            "stain.n.01": ["stain.n.01_1"],
        }
        parsed = [
            ["forall", ["?x", "-", "bowl.n.01"], ["not", ["covered", "?x", "stain.n.01_1"]]]
        ]
        compiled = compile_state(parsed, scope=scope, object_map=object_map)

        # One bowl still covered -> forall fails
        def one_covered(cls, *entities):
            return cls is Covered and entities[0] == "bowl.n.01_2"

        ok, _ = evaluate_state(compiled, one_covered)
        assert ok is False

        # No bowls covered -> forall succeeds
        ok, _ = evaluate_state(compiled, lambda cls, *e: False)
        assert ok is True


# ---------------------------------------------------------------------------
# Full Task Goal Evaluation (cleaning_up_after_a_meal)
# ---------------------------------------------------------------------------


class TestTaskGoalEvaluation:
    """Test check_goal on a real task with entity-name-based callbacks.

    cleaning_up_after_a_meal-0 goal structure:
      0: forall bowls: NOT covered(bowl, stain)
      1: forall plates: NOT covered(plate, stain)
      2: forall cups: NOT covered(cup, stain)
      3: forall hamburgers: inside(hamburger, sack)
      4: ontop(sack, floor)
      5: NOT covered(table, stain)
      6: forall chairs: NOT covered(chair, stain)
    """

    @pytest.fixture()
    def ct(self, kb):
        return kb.get_task("cleaning_up_after_a_meal-0").compile(scene_layout={})

    def test_nothing_satisfied(self, ct):
        ok, results = ct.check_goal(lambda cls, *e: False)
        assert not ok
        assert 3 in results["unsatisfied"] or 4 in results["unsatisfied"]

    def test_all_goal_conditions_met(self, ct):
        def solved_eval(cls, *entities):
            if cls is Covered:
                return False
            if cls is Inside:
                return True
            if cls is OnTop:
                return True
            return False

        ok, results = ct.check_goal(solved_eval)
        assert ok is True
        assert len(results["unsatisfied"]) == 0

    def test_partial_progress(self, ct):
        def partial_eval(cls, *entities):
            if cls is Covered:
                return False
            if cls is OnTop:
                return True
            if cls is Inside:
                return False
            return False

        ok, results = ct.check_goal(partial_eval)
        assert ok is False
        assert 3 in results["unsatisfied"]
        assert 4 in results["satisfied"]

    def test_selective_by_entity_name(self, ct):
        def selective_eval(cls, *entities):
            if cls is Covered:
                return entities[0] == "bowl.n.01_1"
            if cls is Inside:
                return True
            if cls is OnTop:
                return True
            return False

        ok, results = ct.check_goal(selective_eval)
        assert 0 in results["unsatisfied"]
        assert 3 in results["satisfied"]
        assert 4 in results["satisfied"]


# ---------------------------------------------------------------------------
# Sampling with Stubs
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sample_predicate_callback_receives_class(self):
        scope = {"bowl.n.01_1", "table.n.02_1"}
        object_map = {"bowl.n.01": ["bowl.n.01_1"], "table.n.02": ["table.n.02_1"]}
        parsed = [["ontop", "bowl.n.01_1", "table.n.02_1"]]
        compiled = compile_state(parsed, scope=scope, object_map=object_map)

        received = []

        def sample_fn(pred_cls, *args, **kwargs):
            received.append((pred_cls, args, kwargs))
            return True

        # Access the leaf predicate through HEAD -> child
        leaf = compiled[0].children[0]
        assert isinstance(leaf, Predicate)
        leaf.sample(sample_fn, binary_state=True)

        assert len(received) == 1
        assert received[0][0] is OnTop

    def test_sample_returns_callback_result(self):
        scope = {"bowl.n.01_1"}
        object_map = {"bowl.n.01": ["bowl.n.01_1"]}
        parsed = [["cooked", "bowl.n.01_1"]]
        compiled = compile_state(parsed, scope=scope, object_map=object_map)

        leaf = compiled[0].children[0]
        assert leaf.sample(lambda cls, *a, **kw: True, binary_state=True) is True
        assert leaf.sample(lambda cls, *a, **kw: False, binary_state=True) is False


# ---------------------------------------------------------------------------
# Transition Rules
# ---------------------------------------------------------------------------


class TestTransitionRules:
    def test_transition_rules_have_recipes(self, kb):
        typed = [tr for tr in kb.all_transition_rules() if tr.recipe is not None]
        assert len(typed) > 0

    def test_cooking_recipes_have_container_synsets(self, kb):
        cooking = [
            tr for tr in kb.all_transition_rules()
            if isinstance(tr.recipe, CookingRecipe) and tr.recipe.container_synsets
        ]
        assert len(cooking) > 0
        for tr in cooking:
            assert len(tr.recipe.container_synsets) > 0

    def test_machine_recipes_have_machine_synsets(self, kb):
        machines = [
            tr for tr in kb.all_transition_rules()
            if isinstance(tr.recipe, MachineRecipe) and tr.recipe.machine_synsets
        ]
        assert len(machines) > 0

    def test_washer_rule_loaded(self, kb):
        assert kb.washer_rule is not None
        assert len(kb.washer_rule.conditions) > 0


# ---------------------------------------------------------------------------
# Predicate Classes
# ---------------------------------------------------------------------------


class TestPredicates:
    def test_predicate_classes_are_expression_nodes(self):
        scope = {"a", "b"}
        obj_map = {"cat": ["a", "b"]}
        node = OnTop("ontop", scope, ["a", "b"], obj_map, generate_ground_options=False)
        assert isinstance(node, Expression)
        assert isinstance(node, Predicate)
        assert isinstance(node, BinaryPredicate)
        assert isinstance(node, OnTop)

    def test_predicate_arity(self):
        assert UnaryPredicate.arity == 1
        assert BinaryPredicate.arity == 2
        assert Cooked.arity == 1
        assert OnTop.arity == 2

    def test_token_to_predicate_mapping(self):
        assert TOKEN_TO_PREDICATE["ontop"] is OnTop
        assert TOKEN_TO_PREDICATE["cooked"] is Cooked
        assert TOKEN_TO_PREDICATE["inside"] is Inside
        assert TOKEN_TO_PREDICATE["covered"] is Covered
        assert TOKEN_TO_PREDICATE["future"] is Future
        assert TOKEN_TO_PREDICATE["real"] is Real
        assert len(TOKEN_TO_PREDICATE) >= 26  # All domain predicates


# ---------------------------------------------------------------------------
# Wildcard Expansion
# ---------------------------------------------------------------------------


class TestWildcardExpansion:
    def test_expand_wildcards_basic(self, kb):
        """Wildcard task expands when given a scene layout with matching objects."""
        task = kb.get_task("carrying_in_groceries-0")
        assert "*" in task.definition

        layout = {
            "garage": {"car": 1, "floor": 1},
            "kitchen": {"fridge": 2, "floor": 1},
        }
        ct = task.compile(scene_layout=layout)

        assert "electric_refrigerator.n.01_*" not in ct.object_scope
        assert "electric_refrigerator.n.01_1" in ct.object_scope
        assert "electric_refrigerator.n.01_2" in ct.object_scope

    def test_parse_base_scope_strips_wildcards(self, kb):
        """parse_base_scope strips wildcard instances from the scope."""
        task = kb.get_task("carrying_in_groceries-0")
        conditions, scope, inroom = task.parse_base_scope()
        assert "electric_refrigerator.n.01_*" not in scope
        # The non-wildcard instance should still be present
        assert "electric_refrigerator.n.01_1" in scope

    def test_wildcard_expansion_count(self, kb):
        """More objects in the scene means more expanded instances."""
        layout = {
            "garage": {"car": 1, "floor": 1},
            "kitchen": {"fridge": 3, "display_fridge": 1, "floor": 1},
        }
        ct = kb.get_task("carrying_in_groceries-0").compile(scene_layout=layout)
        fridge_instances = [n for n in ct.object_scope if "electric_refrigerator" in n]
        assert len(fridge_instances) == 4

    def test_wildcard_task_compiles_and_evaluates(self, kb):
        """Expanded wildcard task can be evaluated with check_goal."""
        layout = {
            "garage": {"car": 1, "floor": 1},
            "kitchen": {"fridge": 2, "floor": 1},
        }
        ct = kb.get_task("carrying_in_groceries-0").compile(scene_layout=layout)

        ok, results = ct.check_goal(lambda cls, *e: False)
        assert isinstance(ok, bool)
        assert "satisfied" in results and "unsatisfied" in results
