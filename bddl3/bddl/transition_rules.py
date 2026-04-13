"""Transition rule data loading and structured representations.

Transition rules define how objects transform in simulation (e.g. cooking
dough into cookies, mixing ingredients into a drink). The raw JSON files in
``generated_data/transition_map/tm_jsons/`` are loaded by this module and
returned as typed dataclass instances with predicate name strings already
resolved to :class:`~bddl.predicates.Predicate` classes.

Recipe types
------------
- :class:`CookingRecipe` -- Requires a heat source and container.
- :class:`MixingRecipe` -- Requires a mixing tool (no explicit container).
- :class:`MachineRecipe` -- Requires a toggleable machine.
- :class:`SubstanceCookingRecipe` -- Simple substance-to-substance transform.
- :class:`WasherRecipe` -- Maps substances to solvent-based removal conditions.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

from bddl.predicates import TOKEN_TO_PREDICATE, Predicate

_TM_JSON_DIR = pathlib.Path(__file__).parent / "generated_data" / "transition_map" / "tm_jsons"


# ---------------------------------------------------------------------------
# State entry: a (predicate_class, value) pair
# ---------------------------------------------------------------------------

@dataclass
class StateCondition:
    """A single state requirement or assignment.

    Attributes:
        predicate: The :class:`~bddl.predicates.Predicate` subclass.
        value: The desired boolean value (True/False).
    """

    predicate: type[Predicate]
    value: bool


# ---------------------------------------------------------------------------
# Base recipe
# ---------------------------------------------------------------------------

@dataclass
class Recipe:
    """Base class for all transition rule recipes.

    Attributes:
        name: Unique identifier for this recipe.
        input_synsets: Maps input synset strings to required counts.
        output_synsets: Maps output synset strings to produced counts.
        input_states: Maps synset key (single synset for unary, comma-separated
            pair for binary) to a list of :class:`StateCondition` requirements.
            None if no state requirements.
        output_states: Maps synset key to a list of :class:`StateCondition`
            assignments for produced objects. None if no state assignments.
    """

    name: str
    input_synsets: dict[str, int]
    output_synsets: dict[str, int]
    input_states: dict[str, list[StateCondition]] | None = None
    output_states: dict[str, list[StateCondition]] | None = None


# ---------------------------------------------------------------------------
# Concrete recipe types
# ---------------------------------------------------------------------------

@dataclass
class CookingRecipe(Recipe):
    """A recipe that requires a heat source and a container.

    Used by CookingObjectRule and CookingSystemRule in OmniGibson.

    Attributes:
        container_synsets: Allowed container synsets (e.g. ``{"cookie_sheet.n.01"}``).
        heatsource_synsets: Allowed heat source synsets (e.g. ``{"oven.n.01"}``).
        timesteps: Number of heating steps required. None means instantaneous (1).
    """

    container_synsets: set[str] = field(default_factory=set)
    heatsource_synsets: set[str] = field(default_factory=set)
    timesteps: int | None = None


@dataclass
class MixingRecipe(Recipe):
    """A recipe that requires a mixing tool (spoon, whisk, etc.).

    Used by MixingToolRule in OmniGibson. No explicit container field --
    the mixing tool's proximity to a fillable container is checked at runtime.
    """

    pass


@dataclass
class MachineRecipe(Recipe):
    """A recipe that requires a toggleable machine.

    Used by ToggleableMachineRule in OmniGibson.

    Attributes:
        machine_synsets: Allowed machine synsets (e.g. ``{"coffee_maker.n.01"}``).
    """

    machine_synsets: set[str] = field(default_factory=set)


@dataclass
class SubstanceCookingRecipe(Recipe):
    """A simple substance-to-substance cooking transformation.

    Used by CookingPhysicalParticleRule in OmniGibson. Has no container,
    heat source, or state requirements -- just input substances that become
    output substances when heated.
    """

    pass


# ---------------------------------------------------------------------------
# Washer rule (different structure entirely)
# ---------------------------------------------------------------------------

@dataclass
class WasherRecipe:
    """Defines substance removal conditions for a washing machine.

    Attributes:
        conditions: Maps substance synset to removal conditions:
            - ``None`` means "never remove this substance".
            - ``[]`` (empty list) means "always remove this substance".
            - ``["solvent1.n.01", ...]`` means "remove if any listed solvent
              is present in the washer".
    """

    conditions: dict[str, list[str] | None]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_states(states_dict):
    """Convert a raw states dict (from JSON) to use StateCondition instances.

    Args:
        states_dict: ``{synset_key: [[predicate_name, value], ...]}`` or None.

    Returns:
        ``{synset_key: [StateCondition, ...]}`` or None.
    """
    if states_dict is None:
        return None
    result = {}
    for synset_key, entries in states_dict.items():
        result[synset_key] = [
            StateCondition(predicate=TOKEN_TO_PREDICATE[entry[0]], value=entry[1])
            for entry in entries
        ]
    return result


def _parse_recipe(raw, recipe_cls, **extra_fields):
    """Parse a raw JSON recipe dict into a Recipe subclass instance."""
    return recipe_cls(
        name=raw.get("rule_name", raw.get("name", "unnamed")),
        input_synsets=raw.get("input_synsets", {}),
        output_synsets=raw.get("output_synsets", {}),
        input_states=_parse_states(raw.get("input_states")),
        output_states=_parse_states(raw.get("output_states")),
        **extra_fields,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cooking_recipes(json_filename="heat_cook.json"):
    """Load cooking recipes (for CookingObjectRule / CookingSystemRule).

    Returns:
        list[CookingRecipe]: Parsed cooking recipes.
    """
    fpath = _TM_JSON_DIR / json_filename
    if not fpath.exists():
        return []
    with open(fpath, "r") as f:
        raw_list = json.load(f)
    return [
        _parse_recipe(
            raw,
            CookingRecipe,
            container_synsets=set(raw.get("container", {}).keys()),
            heatsource_synsets=set(raw.get("heat_source", {}).keys()),
            timesteps=raw.get("timesteps"),
        )
        for raw in raw_list
    ]


def load_mixing_recipes(json_filename="mixing_stick.json"):
    """Load mixing recipes (for MixingToolRule).

    Returns:
        list[MixingRecipe]: Parsed mixing recipes.
    """
    fpath = _TM_JSON_DIR / json_filename
    if not fpath.exists():
        return []
    with open(fpath, "r") as f:
        raw_list = json.load(f)
    return [_parse_recipe(raw, MixingRecipe) for raw in raw_list]


def load_machine_recipes(json_filename="single_toggleable_machine.json"):
    """Load machine recipes (for ToggleableMachineRule).

    Returns:
        list[MachineRecipe]: Parsed machine recipes.
    """
    fpath = _TM_JSON_DIR / json_filename
    if not fpath.exists():
        return []
    with open(fpath, "r") as f:
        raw_list = json.load(f)
    return [
        _parse_recipe(
            raw,
            MachineRecipe,
            machine_synsets=set(raw.get("machine", {}).keys()),
        )
        for raw in raw_list
    ]


def load_substance_cooking_recipes(*json_filenames):
    """Load substance cooking recipes (for CookingPhysicalParticleRule).

    Args:
        *json_filenames: One or more JSON filenames. Defaults to
            ``"substance_cooking.json"`` and ``"substance_watercooking.json"``.

    Returns:
        list[SubstanceCookingRecipe]: Parsed substance cooking recipes.
    """
    if not json_filenames:
        json_filenames = ("substance_cooking.json", "substance_watercooking.json")
    recipes = []
    for fn in json_filenames:
        fpath = _TM_JSON_DIR / fn
        if not fpath.exists():
            continue
        with open(fpath, "r") as f:
            raw_list = json.load(f)
        recipes.extend(_parse_recipe(raw, SubstanceCookingRecipe) for raw in raw_list)
    return recipes


def load_washer_rule(json_filename="washer.json"):
    """Load washer substance removal conditions.

    Returns:
        WasherRecipe | None: Parsed washer rule, or None if file not found.
    """
    fpath = _TM_JSON_DIR / json_filename
    if not fpath.exists():
        return None
    with open(fpath, "r") as f:
        conditions = json.load(f)
    return WasherRecipe(conditions=conditions)
