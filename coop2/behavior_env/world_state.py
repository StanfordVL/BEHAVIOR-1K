"""L1a: the world model behind the text observation.

Turns the live scene into a sparse, room-grouped entity list with **stable,
type-local ids** (``apple#1``, not ``apple_agveuv_0``). Everything here exists
because an LLM has to read it: crafter's 64x64 dense grid scan has no analogue
worth porting, and raw BEHAVIOR object names are unreadable and unguessable.

Design points that are load-bearing (PORTING_PLAN 3.1 / 4.5):

* **One shared** ``SceneGraphBuilder`` with ``full_obs=True`` and *every* robot
  name. ``full_obs=False`` routes through ``ObjectsInFOVOfRobot``, which needs
  a camera and -- worse -- intersects the FOVs of all robots
  (``objs_to_add &= objs_in_fov``), which is useless for multi-agent. A robot
  missing from ``robot_names`` is deleted from the graph, so teammates would be
  invisible to each other. One builder, not N: at ``full_obs=True`` each builder
  recomputes the same whole-scene predicates, so N builders is N x the cost.
* **Room membership has two sources and they are not interchangeable.**
  ``obj.in_rooms`` is static scene metadata written at load time and never
  updated when an object moves -- carry a cup from the kitchen to the bedroom
  and it still reports ``kitchen_0``. So: fixed furniture uses ``in_rooms``,
  everything movable gets a live ``seg_map.get_room_instance_by_point`` query.
  Robots have no ``in_rooms`` at all and are always point-queried.
* ``get_room_type_by_point`` is **not** used: it indexes a dict with a 0-dim
  tensor and raises ``KeyError`` for any non-boundary point. Room type comes
  from splitting the instance name, as ``behavior_task.py`` does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "BehaviorWorldState",
    "EntityObservation",
    "PredicateFact",
    "SymbolicObservation",
    "room_type_of",
]


def room_type_of(room_instance: Optional[str]) -> Optional[str]:
    """``'living_room_0' -> 'living_room'``. Mirrors behavior_task.py."""
    if not room_instance:
        return None
    return room_instance.rsplit("_", 1)[0]


GRASPING_TRUE = 1


def is_definitely_grasping(state) -> bool:
    """``True`` only for ``IsGraspingState.TRUE`` (an IntEnum with FALSE = -1).

    Duplicated from ``symbolic_contention`` deliberately: the CPU suites load
    each module standalone under a stubbed ``omnigibson``, so a shared import
    would be unimportable there. See that copy for what the truthiness bug cost.
    """
    if state is None or isinstance(state, str):
        return False
    try:
        return int(state) == GRASPING_TRUE
    except (TypeError, ValueError):
        return False


@dataclass
class EntityObservation:
    """One object or robot, as the cognitive layer sees it."""

    entity_id: str  # type-local and stable: "apple#1"
    name: str  # the scene's own name, needed to act on it
    category: str
    rooms: List[str] = field(default_factory=list)
    abilities: List[str] = field(default_factory=list)
    states: Dict[str, bool] = field(default_factory=dict)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    held_by: Optional[str] = None
    is_robot: bool = False
    is_fixed: bool = False

    @property
    def room(self) -> Optional[str]:
        return self.rooms[0] if self.rooms else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.entity_id,
            "name": self.name,
            "category": self.category,
            "rooms": list(self.rooms),
            "states": dict(self.states),
            "position": [round(float(v), 3) for v in self.position],
            "held_by": self.held_by,
            "is_robot": self.is_robot,
        }


@dataclass
class PredicateFact:
    """One relation that currently holds, in entity-id terms."""

    predicate: str
    args: Tuple[str, ...]
    value: bool = True

    def __str__(self) -> str:
        return f"{self.predicate}({', '.join(self.args)})" + ("" if self.value else " = False")


@dataclass
class SymbolicObservation:
    """What one agent is told about the world on one macro-step."""

    agent_id: str
    step: int = 0
    max_steps: Optional[int] = None
    room: Optional[str] = None
    entities: Dict[str, EntityObservation] = field(default_factory=dict)
    facts: List[PredicateFact] = field(default_factory=list)
    goal_status: Optional[Dict[str, Any]] = None
    last_action_id: Optional[str] = None
    last_error: Optional[str] = None

    def by_room(self) -> Dict[Optional[str], List[EntityObservation]]:
        grouped: Dict[Optional[str], List[EntityObservation]] = {}
        for entity in self.entities.values():
            grouped.setdefault(entity.room, []).append(entity)
        for entities in grouped.values():
            entities.sort(key=lambda e: e.entity_id)
        return grouped


#: Binary relations surfaced to the agent. Kept explicit because the cost of a
#: relation scan is (entities^2 x states): OmniGibson's SceneGraphBuilder scans
#: *every* ordered pair in the scene against *every* relative boolean state,
#: which on house_single_floor's 654 objects is 1.59M get_value calls and was
#: measured at 14.9 s per refresh -- 73% of a 150-step episode, against 44 ms
#: for an actual physics tick. Only the room's entities are ever shown to an
#: agent, so only they need relations.
RELATION_STATES = ("OnTop", "Inside", "Under")

#: Unary states surfaced to the agent. These are the ones symbolic_view's
#: _STATE_GATES turns into open/close and toggle_on/toggle_off hints.
UNARY_STATES = ("Open", "ToggledOn")


class BehaviorWorldState:
    """Shared world model. Build once per env; ``step()`` once per macro-step.

    Args:
        env: the OmniGibson environment.
        use_scene_graph: also maintain OmniGibson's ``SceneGraphBuilder``.
            **Off by default**: its all-pairs relation scan costs 14.9 s per
            refresh on a 654-object scene (see RELATION_STATES), and the only
            things read from the graph -- unary states and relations -- are
            computed directly, scoped to the entities actually being shown.
            Turn it on to cross-check the scoped results against upstream's.
            Original note: build relation facts via ``SceneGraphBuilder``. The
            binary kinematic predicates underneath are adjacency ray casts, so
            this is the expensive part; off, only entities and unary states are
            produced.
        exclude_states: passed through to the builder. The default excludes
            ``Touching``/``NextTo``, which are the most expensive of all.
    """

    _taxonomy_cache = None
    _token_cache = None

    def __init__(self, env, use_scene_graph: bool = False, exclude_states=None):
        self.env = env
        self.scene = env.scene
        self.robots = list(env.robots)
        self.robot_names = [robot.name for robot in self.robots]
        self.use_scene_graph = use_scene_graph
        self._exclude_states = exclude_states
        self._builder = None
        self._graph = None
        self._facts_cache_key = None
        self._facts_cache: List[PredicateFact] = []
        self.step_index = 0

        # Stable type-local ids, assigned on first sight and never reused: the
        # LLM refers to "apple#1" across turns, so it must keep meaning the
        # same object even as objects appear or move.
        self._ids: Dict[str, str] = {}
        self._counts: Dict[str, int] = {}
        # Per-agent memory of rooms visited, so an agent keeps knowing about a
        # room it has already been in (PORTING_PLAN 3.1).
        self._seen_rooms: Dict[str, Set[str]] = {name: set() for name in self.robot_names}

    # -- setup ------------------------------------------------------------

    def start(self) -> None:
        """Attach the scene graph builder. Call after ``og.sim.play()``."""
        if not self.use_scene_graph:
            return
        from omnigibson.scene_graphs.graph_builder import SceneGraphBuilder  # noqa: PLC0415

        kwargs = dict(
            robot_names=list(self.robot_names),  # every robot, or teammates vanish
            egocentric=False,
            full_obs=True,  # never the FOV path: it needs a camera and intersects FOVs
            only_true=True,
            merge_parallel_edges=True,
        )
        if self._exclude_states is not None:
            kwargs["exclude_states"] = self._exclude_states
        self._builder = SceneGraphBuilder(**kwargs)
        self._builder.start(self.scene)

    # -- ids ---------------------------------------------------------------

    @staticmethod
    def _taxonomy():
        """The BDDL object taxonomy, loaded once (it parses a JSON hierarchy)."""
        if BehaviorWorldState._taxonomy_cache is None:
            from bddl.object_taxonomy import ObjectTaxonomy  # noqa: PLC0415

            BehaviorWorldState._taxonomy_cache = ObjectTaxonomy()
        return BehaviorWorldState._taxonomy_cache

    def synset_of(self, obj) -> Optional[str]:
        """WordNet synset for @obj's category, e.g. ``apple.n.01``."""
        category = getattr(obj, "category", None)
        if not category:
            return None
        try:
            return self._taxonomy().get_synset_from_category(category)
        except Exception:  # noqa: BLE001 - categories outside the taxonomy
            return None

    def entity_id_of_name(self, name: str) -> Optional[str]:
        """``apple_48`` -> ``apple.n.01_2``, or None if unknown.

        The reverse of entity_id_for, for error text. A primitive raises with
        the *scene* name, because that is all the controller has, and the agent
        has never seen it -- it is told to use only the ids from "You can do:"
        and never to invent one. An outcome that says "Cannot reach apple_48"
        is therefore unusable: the agent cannot tell which of its targets
        failed, or even that the name refers to something it knows.
        """
        return self._ids.get(name)

    def adopt_task_scope(self, task) -> int:
        """Take entity ids straight from a BehaviorTask's object_scope.

        Without this the two namings only *look* alike. ``entity_id_for``
        numbers instances in scene-enumeration order, so
        ``coffee_table.n.01_1`` was whichever coffee table the scene listed
        first, while the activity's goal refers to the one its sampler bound --
        a different table in the same room. The agent then placed both apples
        on "coffee_table.n.01_1" as it had been shown it, and check_goal kept
        reporting the goal unmet because it was looking at the other one. An
        unwinnable task that looks like an agent failure.

        Returns the number of ids adopted.
        """
        scope = getattr(task, "object_scope", None) or {}
        adopted = 0
        for instance_name, entity in scope.items():
            if entity is None:
                continue
            name = getattr(entity, "name", None)
            if not name:
                continue
            self._ids[name] = instance_name
            # Keep the fallback counter past anything the scope already used,
            # so an object outside the scope cannot be handed an id the task
            # has already bound to something else.
            base, _, suffix = instance_name.rpartition("_")
            if base and suffix.isdigit():
                self._counts[base] = max(self._counts.get(base, 0), int(suffix))
            adopted += 1
        return adopted

    def entity_id_for(self, obj) -> str:
        """Stable BDDL-style id for @obj: ``apple.n.01_1``.

        This is BDDL's own instance naming -- an activity definition writes
        ``chlorine__bottle.n.01_1 - chlorine__bottle.n.01`` and its goal
        predicates refer to exactly those strings. Using anything else (an
        earlier version of this used ``apple#1``) means the ids the LLM is
        shown do not match the ids the task's goal is written in, so L1d would
        need a translation layer and every prompt would speak a private
        dialect. Matching now is much cheaper than matching after M9.

        Falls back to the raw category for objects outside the taxonomy --
        robots, and anything added ad hoc -- which keeps ids readable rather
        than raising on a scene the taxonomy does not fully cover.
        """
        name = obj.name
        if name in self._ids:
            return self._ids[name]
        base = self.synset_of(obj) or getattr(obj, "category", None) or type(obj).__name__.lower()
        self._counts[base] = self._counts.get(base, 0) + 1
        self._ids[name] = f"{base}_{self._counts[base]}"
        return self._ids[name]

    # -- rooms -------------------------------------------------------------

    @property
    def seg_map(self):
        return getattr(self.scene, "seg_map", None) or getattr(self.scene, "_seg_map", None)

    def room_at(self, xy) -> Optional[str]:
        seg_map = self.seg_map
        if seg_map is None:
            return None
        try:
            return seg_map.get_room_instance_by_point(xy[:2])
        except Exception:  # noqa: BLE001 - off-map points
            return None

    def rooms_of(self, obj, position=None) -> List[str]:
        """Rooms @obj counts as being in.

        Fixed furniture keeps its static ``in_rooms`` annotation; anything that
        can move is point-queried, because ``in_rooms`` is never updated when an
        object is carried somewhere else.
        """
        fixed = obj in self._fixed_objects()
        in_rooms = list(getattr(obj, "in_rooms", None) or [])
        if fixed and in_rooms:
            return in_rooms
        if position is None:
            position = obj.get_position_orientation()[0]
        room = self.room_at(position)
        return [room] if room else in_rooms

    def room_of_robot(self, robot) -> Optional[str]:
        return self.room_at(robot.get_position_orientation()[0])

    def _fixed_objects(self) -> Set[Any]:
        """The scene's immovable objects, as objects.

        ``scene.fixed_objects`` is a **name -> object** dict, so ``set(...)`` of
        it is a set of *names* and ``obj in`` it is always False. That silently
        made every entity report ``is_fixed=False``, which only surfaced once
        the receptacle test started depending on it.
        """
        fixed = getattr(self.scene, "fixed_objects", None) or {}
        values = fixed.values() if hasattr(fixed, "values") else fixed
        return set(values)

    # -- holders -----------------------------------------------------------

    def held_objects(self) -> Dict[str, str]:
        """``{object name: robot name}`` across every robot and arm.

        The private ``_ag_obj_in_hand`` is read for the *identity* of what is
        held -- it is the only place that information exists, since
        ``is_grasping`` answers yes/no about a candidate you must already name.
        The answer is then confirmed through the public
        ``is_grasping(arm, candidate_obj)``, which is what applies the
        ``grasping_mode == "physical"`` rules.

        Written this way for cost, not taste. Asking ``is_grasping`` about every
        object in the scene is O(robots x arms x objects); house_single_floor has
        654 of them, and this runs inside ``entities()``, which runs per agent on
        every observation refresh. That version made a 4000-tick episode fail to
        finish a single 500-tick NAVIGATE_TO inside its wall-clock budget, and
        the only visible symptom was empty constraint metrics.
        """
        held: Dict[str, str] = {}
        for robot in self.robots:
            in_hand = getattr(robot, "_ag_obj_in_hand", None) or {}
            for arm, candidate in in_hand.items():
                if candidate is None:
                    continue
                try:
                    # == TRUE, not truthiness: IsGraspingState is an IntEnum with
                    # FALSE = -1 and UNKNOWN = 0, so `if state:` accepts a
                    # definite no and rejects "don't know" -- which defeats the
                    # confirmation this call exists to perform. See
                    # symbolic_contention.holder_of for what that cost.
                    confirmed = is_definitely_grasping(
                        robot.is_grasping(arm=arm, candidate_obj=candidate)
                    )
                except Exception:  # noqa: BLE001 - non-manipulation robots
                    confirmed = True
                if confirmed:
                    held[candidate.name] = robot.name
        return held

    # -- stepping ----------------------------------------------------------

    def step(self) -> None:
        """Refresh the world model. One call per macro-step, not per tick."""
        self.step_index += 1
        if self._builder is not None:
            # Both of these are O(scene): step() scans every ordered pair for
            # relations and get_scene_graph() copies the whole graph. Only run
            # when the caller explicitly asked for the upstream graph.
            self._builder.step(self.scene)
            self._graph = self._builder.get_scene_graph()
        for robot in self.robots:
            room = self.room_of_robot(robot)
            if room:
                self._seen_rooms.setdefault(robot.name, set()).add(room)

    # -- observation -------------------------------------------------------

    @staticmethod
    def _state_entry(obj, state_name: str):
        """``obj.states`` entry for @state_name, or None.

        Real ``states`` dicts are keyed by the state *class*; the stubbed
        tests key them by name. Accept both rather than forcing either.
        """
        states = getattr(obj, "states", None)
        if not states:
            return None
        for key, value in states.items():
            if getattr(key, "__name__", None) == state_name or key == state_name:
                return value
        return None

    def _unary_states(self, obj) -> Dict[str, bool]:
        """Boolean unary states for @obj.

        Read from the graph node when a graph is being maintained, otherwise
        straight off the object -- same values, without the all-pairs scan
        that maintaining the graph pays for.
        """
        if self._graph is not None and obj in self._graph.nodes:
            states = self._graph.nodes[obj].get("states", {})
            return {str(k): bool(v) for k, v in states.items()}

        out: Dict[str, bool] = {}
        for state_name in UNARY_STATES:
            entry = self._state_entry(obj, state_name)
            if entry is None:
                continue
            try:
                value = entry.get_value() if hasattr(entry, "get_value") else entry
            except Exception:  # noqa: BLE001 - a state that cannot be evaluated now
                continue
            out[state_name] = bool(value)
        return out

    def entities(self) -> Dict[str, EntityObservation]:
        """Every entity in the scene, keyed by stable id."""
        held = self.held_objects()
        fixed = self._fixed_objects()
        out: Dict[str, EntityObservation] = {}

        for robot in self.robots:
            position = robot.get_position_orientation()[0]
            entity_id = self.entity_id_for(robot)
            room = self.room_of_robot(robot)
            out[entity_id] = EntityObservation(
                entity_id=entity_id,
                name=robot.name,
                category=getattr(robot, "category", "robot") or "robot",
                rooms=[room] if room else [],
                position=tuple(float(v) for v in position),
                is_robot=True,
            )

        for obj in self.scene.objects:
            if obj in self.robots:
                continue
            if getattr(obj, "visual_only", False):
                continue
            position = obj.get_position_orientation()[0]
            entity_id = self.entity_id_for(obj)
            out[entity_id] = EntityObservation(
                entity_id=entity_id,
                name=obj.name,
                category=getattr(obj, "category", "object") or "object",
                rooms=self.rooms_of(obj, position),
                abilities=sorted(getattr(obj, "abilities", None) or []),
                states=self._unary_states(obj),
                position=tuple(float(v) for v in position),
                held_by=held.get(obj.name),
                is_fixed=obj in fixed,
            )
        return out

    @staticmethod
    def _state_to_bddl_token() -> Dict[str, str]:
        """``{OmniGibson state class name: BDDL token}``, e.g. ``OnTop -> ontop``.

        Inverted from ``bddl_utils.PREDICATE_TO_STATE``, which is the official
        BDDL-predicate <-> object-state mapping. The scene graph labels its
        edges with the *state* class name; a goal condition is written with the
        BDDL *token*. Those two happen to look alike for most predicates
        (``OnTop`` / ``ontop``) and differ for others (``Hot`` is
        ``object_states.Heated``, ``Attached`` is ``AttachedTo``), so relying on
        the resemblance would work until exactly the cases that matter.

        With this and the BDDL instance ids, a rendered fact reads
        ``ontop(apple.n.01_1, breakfast_table.n.01_1)`` -- character for
        character what an activity definition writes, which is what M9's
        ``check_goal`` will be comparing against.
        """
        if BehaviorWorldState._token_cache is None:
            from bddl.predicates import TOKEN_TO_PREDICATE  # noqa: PLC0415
            from omnigibson.utils.bddl_utils import PREDICATE_TO_STATE  # noqa: PLC0415

            predicate_to_token = {cls: token for token, cls in TOKEN_TO_PREDICATE.items()}
            BehaviorWorldState._token_cache = {
                state.__name__: predicate_to_token[predicate]
                for predicate, state in PREDICATE_TO_STATE.items()
                if predicate in predicate_to_token
            }
        return BehaviorWorldState._token_cache

    def predicate_token(self, state_name: str) -> str:
        """BDDL token for a scene-graph edge label, or the label unchanged."""
        try:
            return self._state_to_bddl_token().get(str(state_name), str(state_name))
        except Exception:  # noqa: BLE001 - bddl/omnigibson unavailable (stubbed tests)
            return str(state_name)

    def facts(self, entities: Dict[str, EntityObservation]) -> List[PredicateFact]:
        """Relations among @entities, in BDDL tokens over BDDL ids.

        Scoped to the entities passed in -- which is the agent's own room --
        unless a SceneGraphBuilder is being maintained, in which case its edges
        are used instead. The scan is quadratic in the entity count, so the
        difference between "this room" and "the scene" is 47^2 against 654^2.
        """
        if self._graph is None:
            return self._facts_scoped(entities)
        return self._facts_from_graph(entities)

    def _facts_scoped(self, entities: Dict[str, EntityObservation]) -> List[PredicateFact]:
        """Relations computed directly, only among @entities.

        Uses the same ``obj.states[X].get_value(other)`` calls the graph
        builder makes; what is replaced is only its scan policy, which is
        every ordered pair in the scene against every relative boolean state.
        Results are memoised per world refresh, since both agents in a room
        ask for the same set.
        """
        key = (self.step_index, frozenset(entity.name for entity in entities.values()))
        if self._facts_cache_key == key:
            return list(self._facts_cache)

        by_name = self._objects_by_name()
        pairs = []
        for entity in entities.values():
            obj = by_name.get(entity.name)
            if obj is not None:
                pairs.append((entity.entity_id, obj))

        facts: List[PredicateFact] = []
        for source_id, source in pairs:
            for state_name in RELATION_STATES:
                entry = self._state_entry(source, state_name)
                if entry is None or not hasattr(entry, "get_value"):
                    continue
                token = self.predicate_token(state_name)
                for target_id, target in pairs:
                    if target is source:
                        continue
                    try:
                        if entry.get_value(target):
                            facts.append(PredicateFact(token, (source_id, target_id), True))
                    except Exception:  # noqa: BLE001 - state undefined for this pair
                        continue

        facts.sort(key=lambda f: (f.predicate, f.args))
        self._facts_cache_key = key
        self._facts_cache = facts
        return list(facts)

    def _objects_by_name(self) -> Dict[str, Any]:
        """``{name: object}`` over the scene and its robots.

        Built from ``scene.objects`` rather than ``object_registry`` so this
        works when driven without a full scene (the stubbed tests).
        """
        by_name = {getattr(obj, "name", None): obj for obj in self.scene.objects}
        for robot in self.robots:
            by_name[getattr(robot, "name", None)] = robot
        by_name.pop(None, None)
        return by_name

    def relation_holds(self, token: str, source_id: str, target_id: str) -> bool:
        """Is this one binary relation true right now?

        Answers the predicate that was asked for. The alternative -- enumerate
        every true relation in the scene, then test membership -- is quadratic
        in the entity count and was measured at 9.4 s per call on
        house_single_floor, to settle a handful of task goals.
        """
        by_name = self._objects_by_name()
        names = {entity_id: name for name, entity_id in self._ids.items()}
        source = by_name.get(names.get(source_id))
        target = by_name.get(names.get(target_id))
        if source is None or target is None or source is target:
            return False
        for state_name in RELATION_STATES:
            if self.predicate_token(state_name) != token:
                continue
            entry = self._state_entry(source, state_name)
            if entry is None or not hasattr(entry, "get_value"):
                return False
            try:
                return bool(entry.get_value(target))
            except Exception:  # noqa: BLE001 - undefined for this pair
                return False
        return False

    def _facts_from_graph(self, entities: Dict[str, EntityObservation]) -> List[PredicateFact]:
        """Relations read off a maintained SceneGraphBuilder's edges."""
        by_name = {entity.name: entity.entity_id for entity in entities.values()}
        facts: List[PredicateFact] = []
        for source, target, data in self._graph.edges(data=True):
            source_id = by_name.get(getattr(source, "name", None))
            target_id = by_name.get(getattr(target, "name", None))
            if source_id is None or target_id is None:
                continue
            # merge_parallel_edges=True gives {"states": [(name, value), ...]};
            # otherwise each edge carries a single {"value": bool}.
            for name, value in data.get("states", []) or []:
                facts.append(PredicateFact(self.predicate_token(name), (source_id, target_id), bool(value)))
            if "value" in data and not data.get("states"):
                facts.append(
                    PredicateFact(
                        self.predicate_token(data.get("name", "related")),
                        (source_id, target_id),
                        bool(data["value"]),
                    )
                )
        facts.sort(key=lambda f: (f.predicate, f.args))
        return facts

    def observation_for(
        self,
        agent_id: str,
        max_steps: Optional[int] = None,
        env_step: Optional[int] = None,
        include_seen_rooms: bool = False,
        goal_status: Optional[Dict[str, Any]] = None,
        last_action_id: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> SymbolicObservation:
        """The room-level view for one agent.

        **The agent's room only, by default.** Remembered rooms are available
        via ``include_seen_rooms=True`` but are off: an observation that
        accumulates every room ever visited grows without bound over an episode
        and stops being a description of where the agent *is*. The current room
        is also exactly the scope COOP2's spatial constraint is defined on.

        Privileged within that room: no occlusion, no FOV. That matches
        crafter's ``symbolic_view``, which is equally privileged inside its
        window, so the comparison across environments stays honest -- but it
        does need saying out loud in the paper.
        """
        robot = next((r for r in self.robots if r.name == agent_id), None)
        if robot is None:
            raise KeyError(f"No robot named {agent_id!r}; have {self.robot_names}.")
        current_room = self.room_of_robot(robot)
        visible_rooms: Set[str] = {current_room} if current_room else set()
        if include_seen_rooms:
            visible_rooms |= self._seen_rooms.get(agent_id, set())

        everything = self.entities()
        entities = {}
        for entity_id, entity in everything.items():
            if entity.name == agent_id:
                entities[entity_id] = entity
                continue
            if not visible_rooms or not entity.rooms:
                # No seg map, or an entity the map cannot place: showing it is
                # better than hiding it, since hiding makes it unmentionable
                # and therefore unusable.
                entities[entity_id] = entity
                continue
            if set(entity.rooms) & visible_rooms:
                entities[entity_id] = entity

        return SymbolicObservation(
            agent_id=agent_id,
            # env_step, not step_index: max_steps is a budget in env steps, so
            # reporting the macro-step count here made the prompt read
            # "Step 5/1500" when 490 of the 1500 were already spent -- two
            # different units printed as one fraction.
            step=self.step_index if env_step is None else env_step,
            max_steps=max_steps,
            room=current_room,
            entities=entities,
            facts=self.facts(entities),
            goal_status=goal_status,
            last_action_id=last_action_id,
            last_error=last_error,
        )
