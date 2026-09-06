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


class BehaviorWorldState:
    """Shared world model. Build once per env; ``step()`` once per macro-step.

    Args:
        env: the OmniGibson environment.
        use_scene_graph: build relation facts via ``SceneGraphBuilder``. The
            binary kinematic predicates underneath are adjacency ray casts, so
            this is the expensive part; off, only entities and unary states are
            produced.
        exclude_states: passed through to the builder. The default excludes
            ``Touching``/``NextTo``, which are the most expensive of all.
    """

    def __init__(self, env, use_scene_graph: bool = True, exclude_states=None):
        self.env = env
        self.scene = env.scene
        self.robots = list(env.robots)
        self.robot_names = [robot.name for robot in self.robots]
        self.use_scene_graph = use_scene_graph
        self._exclude_states = exclude_states
        self._builder = None
        self._graph = None
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

    def entity_id_for(self, obj) -> str:
        """Stable ``category#n`` id for @obj."""
        name = obj.name
        if name in self._ids:
            return self._ids[name]
        category = getattr(obj, "category", None) or type(obj).__name__.lower()
        self._counts[category] = self._counts.get(category, 0) + 1
        self._ids[name] = f"{category}#{self._counts[category]}"
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
        fixed = obj in set(getattr(self.scene, "fixed_objects", {}) or {})
        in_rooms = list(getattr(obj, "in_rooms", None) or [])
        if fixed and in_rooms:
            return in_rooms
        if position is None:
            position = obj.get_position_orientation()[0]
        room = self.room_at(position)
        return [room] if room else in_rooms

    def room_of_robot(self, robot) -> Optional[str]:
        return self.room_at(robot.get_position_orientation()[0])

    # -- holders -----------------------------------------------------------

    def held_objects(self) -> Dict[str, str]:
        """``{object name: robot name}`` across every robot and arm."""
        held: Dict[str, str] = {}
        for robot in self.robots:
            in_hand = getattr(robot, "_ag_obj_in_hand", None) or {}
            for obj in in_hand.values():
                if obj is not None:
                    held[obj.name] = robot.name
        return held

    # -- stepping ----------------------------------------------------------

    def step(self) -> None:
        """Refresh the world model. One call per macro-step, not per tick."""
        self.step_index += 1
        if self._builder is not None:
            self._builder.step(self.scene)
            self._graph = self._builder.get_scene_graph()
        for robot in self.robots:
            room = self.room_of_robot(robot)
            if room:
                self._seen_rooms.setdefault(robot.name, set()).add(room)

    # -- observation -------------------------------------------------------

    def _unary_states(self, obj) -> Dict[str, bool]:
        """Boolean unary states, read from the graph node when available."""
        if self._graph is not None and obj in self._graph.nodes:
            states = self._graph.nodes[obj].get("states", {})
            return {str(k): bool(v) for k, v in states.items()}
        return {}

    def entities(self) -> Dict[str, EntityObservation]:
        """Every entity in the scene, keyed by stable id."""
        held = self.held_objects()
        fixed = set(getattr(self.scene, "fixed_objects", {}) or {})
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

    def facts(self, entities: Dict[str, EntityObservation]) -> List[PredicateFact]:
        """Relations from the scene graph, translated into entity ids."""
        if self._graph is None:
            return []
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
                facts.append(PredicateFact(str(name), (source_id, target_id), bool(value)))
            if "value" in data and not data.get("states"):
                facts.append(PredicateFact(str(data.get("name", "related")), (source_id, target_id), bool(data["value"])))
        facts.sort(key=lambda f: (f.predicate, f.args))
        return facts

    def observation_for(
        self,
        agent_id: str,
        max_steps: Optional[int] = None,
        include_seen_rooms: bool = True,
        goal_status: Optional[Dict[str, Any]] = None,
        last_action_id: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> SymbolicObservation:
        """The room-level view for one agent.

        Privileged within its rooms: no occlusion, no FOV. That matches
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
            step=self.step_index,
            max_steps=max_steps,
            room=current_room,
            entities=entities,
            facts=self.facts(entities),
            goal_status=goal_status,
            last_action_id=last_action_id,
            last_error=last_error,
        )
