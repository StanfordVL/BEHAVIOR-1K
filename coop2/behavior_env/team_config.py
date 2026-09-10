"""Who is in the scene, what kind of robot each one is, and who they work with.

Reads one JSON file. Everything an experiment needs to know about the *team* --
robot models, where each robot starts, and how robots are grouped -- lives there
rather than in CLI flags, because the three are not independent: a team is a set
of named robots, and a name only means something once that robot has a model and
a place to stand.

Shape::

    {
      "robots": [
        {"name": "agent_0", "model": "R1",    "position": [-9.1, -1.8], "team": "alpha"},
        {"name": "agent_1", "model": "Tiago", "room": "living_room_0",  "team": "alpha"},
        {"name": "agent_2", "model": "R1",    "room": "kitchen_0",      "team": "bravo"}
      ]
    }

``position`` and ``room`` are two ways to say where a robot starts and exactly
one of them is required per robot:

* ``position`` is exact -- ``[x, y]`` or ``[x, y, z]``, world coordinates. Use it
  to reproduce a layout precisely, or to put a robot somewhere the room-scoped
  sampler would never choose.
* ``room`` is a room *instance* ("living_room_0", not "living_room"): the robot
  is placed by the usual sampler inside that room, mutually separated from
  everyone else. Use it when the exact spot does not matter.

Teams may be given per robot (``"team": "alpha"``) or as an explicit map::

      "teams": {"alpha": ["agent_0", "agent_1"], "bravo": ["agent_2"]}

Both forms are accepted and they must agree if both are present. A robot with no
team at all lands in a team of its own, which is what makes the single-agent and
one-LLM-per-robot topologies a special case of the team topology rather than a
separate code path.

A robot that does not ship with BEHAVIOR is declared the same way, plus the
config that describes it::

      {"name": "agent_3", "model": "MyRobot", "room": "kitchen_0",
       "config": "/path/to/myrobot_primitives.yaml"}

That file needs one thing: a ``robots[0]`` entry with the controller stack the
symbolic primitives require (HolonomicBaseJointController for the base,
JointControllers for trunk/arms/grippers, all absolute position mode). Nothing
else in coop2 asks what kind of robot it is driving. Equivalently, call
:func:`register_robot_model` once at import time and then name the model as
usual.

Validation is strict and up front on purpose. Every field here ends up either in
an OmniGibson robot config or in a placement call, and both of those fail late
and confusingly: a bad model name surfaces as a missing YAML halfway through
scene load, and a bad room name surfaces as a robot standing in the wrong place
with no error at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BUILTIN_ROBOT_MODELS",
    "RobotSpec",
    "TeamLayout",
    "known_robot_models",
    "load_team_layout",
    "parse_team_layout",
    "register_robot_model",
    "robot_model_config_path",
]

#: Robot models that ship with OmniGibson and are known to work here. The
#: symbolic primitives require the controller stack in
#: ``<model>_primitives.yaml`` -- a HolonomicBaseJointController for the base and
#: JointControllers for trunk/arms/grippers, all absolute position mode -- and
#: these three are the only ones that ship it. Upstream says the same in its own
#: warning: the primitives "only work with Tiago and R1".
#:
#: R1Pro is listed because the config exists, but note it has never been run
#: here: it was avoided while cuRobo was in use (cuRobo drops the DEFAULT
#: embodiment at cuda capability (12, 0), so ``update_obstacles`` raises a
#: KeyError on RTX-50 class cards). cuRobo is gone now, so the objection may be
#: gone with it -- but "may be" is not "is".
BUILTIN_ROBOT_MODELS = ("R1", "R1Pro", "Tiago")

#: ``{lowercase name: (canonical name, config path or None)}``. A path of None
#: means "resolve it from OmniGibson's example configs at build time", which is
#: how the built-ins work; a real path is what a **non-native robot** supplies.
#:
#: This is a registry rather than a fixed tuple because importing robots that do
#: not ship with BEHAVIOR is planned (user, 2026-09-10). Such a robot needs
#: exactly one thing from us -- a primitives-style config with the controller
#: stack above -- so that is the whole extension point: either name the file in
#: the layout JSON (``"config": "/path/to/myrobot_primitives.yaml"``) or call
#: :func:`register_robot_model` once at import time. Nothing else in coop2 asks
#: what kind of robot it is driving; ``robot.q_to_action`` and the base joints
#: are the only interface, and they come from that config.
_ROBOT_MODELS: Dict[str, Tuple[str, Optional[str]]] = {
    model.lower(): (model, None) for model in BUILTIN_ROBOT_MODELS
}


def register_robot_model(name: str, config_path: Optional[str] = None) -> str:
    """Make @name usable as a ``model`` in a team layout. Returns the canonical name.

    Args:
        name: what layouts will call it, e.g. ``"MyRobot"``. Matched
            case-insensitively.
        config_path: a primitives-style YAML whose ``robots[0]`` entry carries
            the controller stack the symbolic primitives need. ``None`` means
            OmniGibson ships ``<name>_primitives.yaml`` in its example configs,
            which is only true of the built-ins.

    Registering the same name twice with different paths raises: two robots
    silently sharing a name would differ only in which one was imported last.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"robot model name must be a non-empty string, got {name!r}")
    key = name.strip().lower()
    if config_path is not None:
        if not isinstance(config_path, str) or not config_path.strip():
            raise ValueError(f"{name}: config_path must be a non-empty string or None")
        config_path = os.path.abspath(os.path.expanduser(config_path.strip()))
    existing = _ROBOT_MODELS.get(key)
    if existing is not None and existing[1] != config_path:
        raise ValueError(
            f"robot model {name!r} is already registered as {existing[0]!r} with config "
            f"{existing[1]!r}; refusing to redefine it as {config_path!r}"
        )
    _ROBOT_MODELS[key] = (name.strip(), config_path)
    return name.strip()


def known_robot_models() -> Tuple[str, ...]:
    """Every model a layout may name right now, built-in and registered."""
    return tuple(canonical for canonical, _ in _ROBOT_MODELS.values())


def robot_model_config_path(model: str) -> Optional[str]:
    """The primitives config registered for @model, or None to resolve by name."""
    entry = _ROBOT_MODELS.get(model.strip().lower())
    if entry is None:
        raise KeyError(f"unknown robot model {model!r}; have {list(known_robot_models())}")
    return entry[1]


@dataclass(frozen=True)
class RobotSpec:
    """One robot: what it is, where it starts, and whose team it is on."""

    name: str
    model: str
    team: str
    #: ``(x, y, z)`` world coordinates, or None when ``room`` decides.
    position: Optional[Tuple[float, float, float]] = None
    #: Room *instance* to sample a pose in, or None when ``position`` decides.
    room: Optional[str] = None

    @property
    def placed_explicitly(self) -> bool:
        return self.position is not None

    @property
    def config_path(self) -> Optional[str]:
        """The primitives YAML for this model, or None to resolve it by name.

        Not stored on the spec: the registry is the single source, so a model
        registered after the layout was parsed still resolves correctly.
        """
        return robot_model_config_path(self.model)

    def config_name(self) -> str:
        """The lowercase model name, which is what OmniGibson's config wants."""
        return self.model.lower()


@dataclass(frozen=True)
class TeamLayout:
    """Every robot in the scene, and the teams they are grouped into."""

    robots: Tuple[RobotSpec, ...]
    #: ``{team name: (agent names, in the order given)}``.
    teams: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    @property
    def agent_names(self) -> Tuple[str, ...]:
        return tuple(spec.name for spec in self.robots)

    @property
    def n_agents(self) -> int:
        return len(self.robots)

    def spec_for(self, name: str) -> RobotSpec:
        for spec in self.robots:
            if spec.name == name:
                return spec
        raise KeyError(f"no robot named {name!r}; have {list(self.agent_names)}")

    def team_of(self, name: str) -> str:
        return self.spec_for(name).team

    def teammates_of(self, name: str) -> Tuple[str, ...]:
        """Every agent on the same team, **including** @name itself.

        Including it is deliberate: callers use this to build the set a team
        brain reasons over, and a brain that plans for everyone but the agent
        that triggered it would be a silent hole.
        """
        return self.teams[self.team_of(name)]

    @property
    def models(self) -> Tuple[str, ...]:
        return tuple(spec.model for spec in self.robots)

    @property
    def is_heterogeneous(self) -> bool:
        return len(set(self.models)) > 1

    def describe(self) -> str:
        lines = [f"{self.n_agents} robots in {len(self.teams)} team(s):"]
        for team, members in self.teams.items():
            lines.append(f"  {team}: {', '.join(members)}")
        for spec in self.robots:
            where = (
                f"at ({spec.position[0]:.2f}, {spec.position[1]:.2f})"
                if spec.placed_explicitly
                else f"sampled in {spec.room}"
            )
            lines.append(f"    {spec.name:<10} {spec.model:<7} {where}")
        return "\n".join(lines)


def _normalise_model(raw: Any, where: str, config: Any = None) -> str:
    """Canonical model name, registering a non-native robot on the way if needed."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{where}: 'model' must be a non-empty string, got {raw!r}")
    key = raw.strip().lower()

    if config is not None:
        # A robot that does not ship with BEHAVIOR. The layout names its
        # primitives config and we register it for the rest of the process, so
        # everything downstream treats it exactly like a built-in.
        if not isinstance(config, str) or not config.strip():
            raise ValueError(f"{where}: 'config' must be a path to a primitives YAML")
        path = os.path.abspath(os.path.expanduser(config.strip()))
        if not os.path.exists(path):
            raise ValueError(
                f"{where}: 'config' points at {path}, which does not exist. It must be a "
                "primitives-style YAML whose robots[0] entry carries the controller stack "
                "the symbolic primitives need."
            )
        return register_robot_model(raw.strip(), path)

    entry = _ROBOT_MODELS.get(key)
    if entry is None:
        raise ValueError(
            f"{where}: unknown robot model {raw!r}. Known: "
            f"{', '.join(known_robot_models())}. The symbolic primitives need the "
            "controller stack from a <model>_primitives.yaml; for a robot that does "
            "not ship with BEHAVIOR, add \"config\": \"/path/to/that.yaml\" to this "
            "entry, or call register_robot_model() before loading the layout."
        )
    return entry[0]


def _normalise_position(raw: Any, where: str) -> Tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) not in (2, 3):
        raise ValueError(
            f"{where}: 'position' must be [x, y] or [x, y, z], got {raw!r}"
        )
    try:
        values = [float(v) for v in raw]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where}: 'position' must be numbers, got {raw!r}") from error
    if len(values) == 2:
        # Spawn height, not floor height. place_robots uses the same default and
        # then corrects: the robot is dropped a few centimetres and settles.
        values.append(0.05)
    return (values[0], values[1], values[2])


def _parse_robot(entry: Any, index: int) -> RobotSpec:
    where = f"robots[{index}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: must be an object, got {type(entry).__name__}")

    name = entry.get("name") or f"agent_{index}"
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where}: 'name' must be a non-empty string, got {name!r}")

    model = _normalise_model(entry.get("model", "R1"), where, entry.get("config"))

    has_position = entry.get("position") is not None
    has_room = entry.get("room") is not None
    if has_position and has_room:
        raise ValueError(
            f"{where} ({name}): give either 'position' or 'room', not both -- "
            "they are two answers to the same question and there is no sensible "
            "way to honour both."
        )
    if not has_position and not has_room:
        raise ValueError(
            f"{where} ({name}): needs 'position' ([x, y] world coordinates) or "
            "'room' (a room instance such as 'living_room_0')."
        )

    position = _normalise_position(entry["position"], where) if has_position else None
    room = None
    if has_room:
        room = entry["room"]
        if not isinstance(room, str) or not room.strip():
            raise ValueError(f"{where} ({name}): 'room' must be a non-empty string")
        room = room.strip()

    team = entry.get("team")
    if team is not None and (not isinstance(team, str) or not team.strip()):
        raise ValueError(f"{where} ({name}): 'team' must be a non-empty string")

    return RobotSpec(
        name=name.strip(),
        model=model,
        team=(team.strip() if team else ""),  # resolved against "teams" below
        position=position,
        room=room,
    )


def _resolve_teams(
    specs: List[RobotSpec], declared: Optional[Dict[str, Any]]
) -> Tuple[Tuple[RobotSpec, ...], Dict[str, Tuple[str, ...]]]:
    """Reconcile per-robot ``team`` fields with an explicit ``teams`` map."""
    by_name = {spec.name: spec for spec in specs}

    from_map: Dict[str, str] = {}
    if declared is not None:
        if not isinstance(declared, dict):
            raise ValueError("'teams' must be an object of {team: [agent names]}")
        for team, members in declared.items():
            if not isinstance(members, (list, tuple)):
                raise ValueError(f"teams[{team!r}] must be a list of agent names")
            for member in members:
                if member not in by_name:
                    raise ValueError(
                        f"teams[{team!r}] names {member!r}, which is not in 'robots' "
                        f"({sorted(by_name)})"
                    )
                if member in from_map:
                    raise ValueError(
                        f"{member!r} appears in two teams: {from_map[member]!r} and {team!r}"
                    )
                from_map[member] = team

    resolved: List[RobotSpec] = []
    for spec in specs:
        mapped = from_map.get(spec.name)
        if spec.team and mapped and spec.team != mapped:
            raise ValueError(
                f"{spec.name}: 'team' says {spec.team!r} but the 'teams' map puts it "
                f"in {mapped!r}. Give one or the other, or make them agree."
            )
        # A robot with no team gets one of its own, so a per-agent topology is a
        # team topology with teams of one rather than a separate code path.
        team = spec.team or mapped or spec.name
        resolved.append(
            RobotSpec(
                name=spec.name, model=spec.model, team=team,
                position=spec.position, room=spec.room,
            )
        )

    teams: Dict[str, List[str]] = {}
    for spec in resolved:
        teams.setdefault(spec.team, []).append(spec.name)
    return tuple(resolved), {team: tuple(members) for team, members in teams.items()}


def parse_team_layout(payload: Dict[str, Any]) -> TeamLayout:
    """Validate an already-loaded JSON object into a :class:`TeamLayout`."""
    if not isinstance(payload, dict):
        raise ValueError(f"team layout must be a JSON object, got {type(payload).__name__}")

    entries = payload.get("robots")
    if not isinstance(entries, list) or not entries:
        raise ValueError("team layout needs a non-empty 'robots' list")

    specs = [_parse_robot(entry, index) for index, entry in enumerate(entries)]

    names = [spec.name for spec in specs]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"duplicate robot names {sorted(duplicates)}. Names are the key of the "
            "action dict, the observation dict and every metric, so they must be unique."
        )

    robots, teams = _resolve_teams(specs, payload.get("teams"))
    return TeamLayout(robots=robots, teams=teams)


def load_team_layout(path: str) -> TeamLayout:
    """Read and validate a team layout JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"no team layout at {path}")
    with open(path, "r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except ValueError as error:
            raise ValueError(f"{path} is not valid JSON: {error}") from error
    try:
        return parse_team_layout(payload)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def homogeneous_layout(
    n_agents: int,
    model: str = "R1",
    room: Optional[str] = None,
    team_size: Optional[int] = None,
) -> TeamLayout:
    """The layout the old ``--agents N`` flag meant, as a :class:`TeamLayout`.

    Keeps the CLI working without a second code path: ``--agents 9`` is nine R1s
    sampled in the task room. ``team_size`` groups them into teams of that size
    (the remainder forms a smaller last team); without it every robot is its own
    team, which is the one-LLM-per-robot behaviour.
    """
    payload: Dict[str, Any] = {"robots": []}
    for index in range(n_agents):
        entry: Dict[str, Any] = {"name": f"agent_{index}", "model": model}
        # room=None is legal here and means "the env picks the room"; the loader
        # requires one of the two, so fill in the placeholder the env resolves.
        entry["room"] = room if room else "*"
        if team_size and team_size > 0:
            entry["team"] = f"team_{index // team_size}"
        payload["robots"].append(entry)
    return parse_team_layout(payload)
