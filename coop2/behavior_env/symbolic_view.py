"""L1b: the world model rendered as text, plus the legal-action catalogue.

``target_hints`` is the single most important thing in this file. It is the
**only** way the LLM learns which ids exist and which primitives apply to them
(PORTING_PLAN 3.1). If a target is missing here the LLM cannot act on it; if a
non-target appears here the LLM will try it and burn a decision on a
precondition failure.

The vocabulary is the symbolic primitive set. Unlike
``StarterSemanticActionPrimitives`` -- where OPEN / CLOSE / TOGGLE_ON /
TOGGLE_OFF all ``raise NotImplementedError`` -- the symbolic set implements all
of them, which is why dependency constraints ("open the fridge before taking
what is inside") are expressible here at all.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from coop2.behavior_env.world_state import EntityObservation, SymbolicObservation, room_type_of

__all__ = ["ActionHint", "render_symbolic_view", "target_hints"]

#: Scenery: in the world model, out of the prompt. These are the synsets whose
#: subtrees the taxonomy has no manipulation abilities for -- structure and
#: fixtures. Measured on house_single_floor, one corridor yields 218 entities
#: and 132 legal (primitive, target) pairs (78 walls, 24 shelves, 20 switches,
#: 16 downlights, 14 paintings), which is not a prompt, it is a haystack.
#:
#: Matched by synset ancestry rather than a category-name list, so a category
#: nobody thought of still resolves correctly. Filtering happens in L1b, **not**
#: L1a: the world model stays complete for task evaluation and only the prompt
#: is pruned.
STRUCTURAL_SYNSETS = (
    "wall.n.01",
    "floor.n.01",
    "ceiling.n.01",
    "roof.n.01",
    "window.n.01",
    "door.n.01",
    "lamp.n.02",
    "light_source.n.01",
    "picture.n.01",
    "mirror.n.01",
    "rug.n.01",
)

#: Abilities the taxonomy annotates that make an object a valid place target.
#: Replaces a hand-written list of ~25 category names, which silently missed
#: anything not on it -- the annotations are the dataset's own answer.
RECEPTACLE_ABILITIES = ("fillable", "openable")


def _taxonomy():
    from bddl.object_taxonomy import ObjectTaxonomy  # noqa: PLC0415

    global _TAXONOMY
    if _TAXONOMY is None:
        _TAXONOMY = ObjectTaxonomy()
    return _TAXONOMY


_TAXONOMY = None


def _synset_of(entity):
    """Synset from the entity id, which is BDDL-shaped: ``apple.n.01_1``."""
    base = entity.entity_id.rsplit("_", 1)[0]
    return base if ".n." in base else None


def is_structural(entity) -> bool:
    """Scenery -- present in the world model, absent from the prompt."""
    synset = _synset_of(entity)
    if synset is None:
        return False
    if synset in STRUCTURAL_SYNSETS:
        return True
    try:
        taxonomy = _taxonomy()
        return any(taxonomy.is_descendant(synset, ancestor) for ancestor in STRUCTURAL_SYNSETS)
    except Exception:  # noqa: BLE001 - synsets outside the taxonomy
        return False


def is_receptacle(entity) -> bool:
    """Can something be placed on or in this?

    Reads the taxonomy's own ability annotations, carried on the loaded object
    as ``obj.abilities`` and copied into the observation.
    """
    if set(entity.abilities) & set(RECEPTACLE_ABILITIES):
        return True
    # A fixed, non-structural object with a surface is a plausible support even
    # without a fillable/openable annotation -- tables and countertops have
    # neither.
    return entity.is_fixed and not is_structural(entity)


#: Unary states that gate a primitive, and the primitive pair they gate.
_STATE_GATES = {
    "Open": ("open", "close"),
    "ToggledOn": ("toggle_on", "toggle_off"),
}


class ActionHint:
    """One legal (primitive, target) pair, with the reason it is legal."""

    __slots__ = ("primitive", "target_id", "target_name", "note")

    def __init__(self, primitive: str, target_id: str, target_name: str, note: str = ""):
        self.primitive = primitive
        self.target_id = target_id
        self.target_name = target_name
        self.note = note

    def __repr__(self) -> str:
        return f"ActionHint({self.primitive}, {self.target_id})"

    def to_dict(self) -> Dict[str, str]:
        return {
            "primitive": self.primitive,
            "target": self.target_id,
            "target_name": self.target_name,
            "note": self.note,
        }


def _holding(observation: SymbolicObservation) -> Optional[EntityObservation]:
    for entity in observation.entities.values():
        if entity.held_by == observation.agent_id:
            return entity
    return None


def target_hints(
    observation: SymbolicObservation,
    interaction_radius: Optional[float] = None,
    include_structural: bool = False,
) -> List[ActionHint]:
    """Every primitive the agent could legally issue right now.

    Proximity is reported, not enforced, when @interaction_radius is given: an
    out-of-range target still yields NAVIGATE_TO, and the note says why the
    manipulation primitives are absent. Silently dropping the target instead
    would leave the LLM unable to discover that navigating fixes it.
    """
    me = observation.entities.get(_agent_entity_id(observation))
    held = _holding(observation)
    hints: List[ActionHint] = []

    for entity in sorted(observation.entities.values(), key=lambda e: e.entity_id):
        if entity.is_robot:
            continue
        if not include_structural and is_structural(entity):
            continue

        distance = None
        if me is not None:
            distance = ((entity.position[0] - me.position[0]) ** 2 + (entity.position[1] - me.position[1]) ** 2) ** 0.5
        in_range = interaction_radius is None or distance is None or distance <= interaction_radius
        far_note = "" if in_range else f"too far ({distance:.1f} m) -- navigate_to first"

        hints.append(
            ActionHint("navigate_to", entity.entity_id, entity.name, "" if in_range else f"{distance:.1f} m away")
        )
        if not in_range:
            continue

        if entity.held_by is None and held is None and not entity.is_fixed:
            hints.append(ActionHint("grasp", entity.entity_id, entity.name, far_note))
        elif entity.held_by not in (None, observation.agent_id):
            # Surfaced deliberately: "who holds what" is the cross-agent signal
            # the topology layer is measured on, so the LLM must be able to see
            # it rather than discover it through a failure.
            hints.append(
                ActionHint("blocked", entity.entity_id, entity.name, f"held by {entity.held_by}")
            )

        if held is not None and entity.entity_id != held.entity_id and is_receptacle(entity):
            hints.append(ActionHint("place_on_top", entity.entity_id, entity.name, f"places {held.entity_id}"))
            if "openable" in entity.abilities or "fillable" in entity.abilities:
                hints.append(ActionHint("place_inside", entity.entity_id, entity.name, f"places {held.entity_id}"))

        for state, (on_verb, off_verb) in _STATE_GATES.items():
            if state not in entity.states:
                continue
            if entity.states[state]:
                hints.append(ActionHint(off_verb, entity.entity_id, entity.name, f"{state} is True"))
            else:
                hints.append(ActionHint(on_verb, entity.entity_id, entity.name, f"{state} is False"))

    if held is not None:
        hints.append(ActionHint("release", held.entity_id, held.name, "drops what you hold"))
    return hints


def _agent_entity_id(observation: SymbolicObservation) -> Optional[str]:
    for entity_id, entity in observation.entities.items():
        if entity.name == observation.agent_id:
            return entity_id
    return None


def render_symbolic_view(
    observation: SymbolicObservation,
    interaction_radius: Optional[float] = None,
    max_facts: int = 25,
    include_structural: bool = False,
) -> str:
    """The scene as prose for the prompt.

    Grouped by room rather than listed flat: the room is what COOP2's spatial
    constraint is defined on, and it is how a person would describe a house.
    """
    lines: List[str] = []
    header = f"Step {observation.step}"
    if observation.max_steps:
        header += f"/{observation.max_steps}"
    room_type = room_type_of(observation.room)
    header += f" | you are {observation.agent_id}"
    header += f" in {observation.room}" if observation.room else " (room unknown)"
    if room_type and room_type != observation.room:
        header += f" (a {room_type.replace('_', ' ')})"
    lines.append(header)

    held = _holding(observation)
    lines.append(f"Holding: {held.entity_id} ({held.category})" if held else "Holding: nothing")

    for room, entities in sorted(observation.by_room().items(), key=lambda kv: (kv[0] is None, kv[0] or "")):
        visible = [e for e in entities if not e.is_robot or e.name != observation.agent_id]
        if not include_structural:
            visible = [e for e in visible if not is_structural(e)]
        if not visible:
            continue
        lines.append(f"\n{room or 'elsewhere'}:")
        for entity in visible:
            bits = [entity.entity_id]
            if entity.is_robot:
                bits.append("(teammate)")
            if entity.held_by:
                bits.append(f"held by {entity.held_by}")
            active = [name for name, value in sorted(entity.states.items()) if value]
            if active:
                bits.append(", ".join(active))
            lines.append("  - " + "  ".join(bits))

    shown_ids = {e.entity_id for e in observation.entities.values() if include_structural or not is_structural(e)}
    facts = [f for f in observation.facts if all(arg in shown_ids for arg in f.args)]
    if facts:
        lines.append("\nRelations:")
        for fact in facts[:max_facts]:
            lines.append(f"  {fact}")
        if len(facts) > max_facts:
            lines.append(f"  ... and {len(facts) - max_facts} more")

    hints = target_hints(observation, interaction_radius=interaction_radius, include_structural=include_structural)
    if hints:
        lines.append("\nYou can do:")
        by_target: Dict[str, List[ActionHint]] = {}
        for hint in hints:
            by_target.setdefault(hint.target_id, []).append(hint)
        for target_id, target_hint_list in sorted(by_target.items()):
            verbs = ", ".join(sorted({h.primitive for h in target_hint_list}))
            note = next((h.note for h in target_hint_list if h.note), "")
            lines.append(f"  {target_id}: {verbs}" + (f"   [{note}]" if note else ""))

    if observation.last_error:
        lines.append(f"\nLast action failed: {observation.last_error}")
    if observation.goal_status:
        lines.append(f"\nGoal: {observation.goal_status}")
    return "\n".join(lines)
