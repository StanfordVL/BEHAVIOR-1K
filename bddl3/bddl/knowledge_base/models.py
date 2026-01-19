from dataclasses import dataclass, field
from functools import cached_property, cache
import itertools
import json
import networkx as nx
from typing import Dict, Optional, Set, List, Tuple
from bddl.object_taxonomy import ObjectTaxonomy
from bddl.knowledge_base.orm import (
    Model,
    ManyToOne,
    ManyToMany,
    OneToMany,
    ManyToOneField,
    ManyToManyField,
    OneToManyField,
    UUIDField,
)
from bddl.knowledge_base.utils import SynsetState
from collections import defaultdict


ROOM_TYPE_CHOICES = [
    ("bar", "bar"),
    ("bathroom", "bathroom"),
    ("bedroom", "bedroom"),
    ("biology lab", "biology_lab"),
    ("break room", "break_room"),
    ("chemistry lab", "chemistry_lab"),
    ("classroom", "classroom"),
    ("computer lab", "computer_lab"),
    ("conference hall", "conference_hall"),
    ("copy room", "copy_room"),
    ("corridor", "corridor"),
    ("dining room", "dining_room"),
    ("empty room", "empty_room"),
    ("grocery store", "grocery_store"),
    ("gym", "gym"),
    ("hammam", "hammam"),
    ("infirmary", "infirmary"),
    ("kitchen", "kitchen"),
    ("lobby", "lobby"),
    ("locker room", "locker_room"),
    ("meeting room", "meeting_room"),
    ("phone room", "phone_room"),
    ("private office", "private_office"),
    ("sauna", "sauna"),
    ("shared office", "shared_office"),
    ("spa", "spa"),
    ("entryway", "entryway"),
    ("television room", "television_room"),
    ("utility room", "utility_room"),
    ("garage", "garage"),
    ("closet", "closet"),
    ("childs room", "childs_room"),
    ("exercise room", "exercise_room"),
    ("garden", "garden"),
    ("living room", "living_room"),
    ("pantry room", "pantry_room"),
    ("playroom", "playroom"),
    ("staircase", "staircase"),
    ("storage room", "storage_room"),
]

OBJECT_TAXONOMY = ObjectTaxonomy()


@dataclass(eq=False, order=False)
class Property(Model):
    name: str
    parameters: str
    id: str = UUIDField()
    synset_fk: ManyToOne = ManyToOneField("Synset", "properties")

    class Meta:
        pk = "id"
        unique_together = ("synset", "name")
        ordering = ["name"]

    def __str__(self):
        return self.synset.name + "-" + self.name


@dataclass(eq=False, order=False)
class MetaLink(Model):
    name: str
    on_objects_fk: ManyToMany = ManyToManyField("Object", "meta_links")

    class Meta:
        pk = "name"


@dataclass(eq=False, order=False)
class AttachmentPair(Model):
    name: str
    female_objects_fk: ManyToMany = ManyToManyField("Object", "female_attachment_pairs")
    male_objects_fk: ManyToMany = ManyToManyField("Object", "male_attachment_pairs")

    class Meta:
        pk = "name"

    @classmethod
    def view_error_missing_objects(cls):
        """Attachment pairs that only have objects on one side"""
        return [
            x
            for x in cls.all_objects()
            if len(x.female_objects) == 0 or len(x.male_objects) == 0
        ]


@dataclass(eq=False, order=False)
class Predicate(Model):
    name: str
    synsets_fk: ManyToMany = ManyToManyField("Synset", "used_in_predicates")
    tasks_fk: ManyToMany = ManyToManyField("Task", "uses_predicates")

    class Meta:
        pk = "name"


@dataclass(eq=False, order=False)
class Scene(Model):
    name: str

    rooms_fk: OneToMany = OneToManyField("Room", "scene")

    @cached_property
    def room_count(self):
        return len(self.rooms)

    @cached_property
    def object_count(self):
        return sum(
            roomobject.count
            for room in self.rooms
            for roomobject in room.non_clutter_roomobjects  # TODO: Should we include clutter objects?
        )

    class Meta:
        pk = "name"
        ordering = ["name"]

    @classmethod
    def view_challenge(cls):
        """Scenes that are included in the NeurIPS 2025 BEHAVIOR Challenge"""
        return [
            x
            for x in cls.all_objects()
            if x.name
            in [
                "house_single_floor",
                "house_double_floor_lower",
                "house_double_floor_upper",
            ]
        ]


@dataclass(eq=False, order=False)
class ParticleSystem(Model):
    name: str
    parameters: Optional[str] = None

    # the synset that the category belongs to
    synset_fk: ManyToOne = ManyToOneField("Synset", "particle_systems")

    # the objects that belong to this particle system as particles
    particles_fk: OneToMany = OneToManyField("Object", "particle_system")

    def __str__(self):
        return self.name

    class Meta:
        pk = "name"
        ordering = ["name"]

    @cache
    def matching_synset(self, synset) -> bool:
        return synset.name in self.matching_synsets

    @cached_property
    def matching_synsets(self) -> Set["Synset"]:
        if not self.synset:
            return set()
        return {anc.name for anc in self.synset.ancestors} | {self.synset.name}

    @cached_property
    def state(self) -> SynsetState:
        # A particle system is ready if it doesn't need particles or if it has any ready particles.
        if self.synset.is_liquid or (
            len(self.particles) > 0
            and any(particle.ready for particle in self.particles)
        ):
            return SynsetState.MATCHED
        elif len(self.particles) == 0:
            return SynsetState.UNMATCHED
        else:
            return SynsetState.PLANNED

    @classmethod
    def view_error_mapped_to_non_leaf_synsets(cls):
        """Particle systems mapped to non-leaf synsets"""
        return [x for x in cls.all_objects() if len(x.synset.children) > 0]

    @classmethod
    def view_error_mapped_to_non_substance_synsets(cls):
        """Particle systems mapped to non-substance synsets"""
        return [x for x in cls.all_objects() if not x.synset.is_substance]

    @classmethod
    def view_error_missing_particle(cls):
        """Particle systems that are missing particles"""
        return [
            x
            for x in cls.all_objects()
            if not x.synset.is_liquid and len(x.particles) == 0
        ]

    @classmethod
    def view_error_missing_params(cls):
        """Particle systems that are missing parameters"""
        return [x for x in cls.all_objects() if not x.parameters]


@dataclass(eq=False, order=False)
class Category(Model):
    name: str

    # the synset that the category belongs to
    synset_fk: ManyToOne = ManyToOneField("Synset", "categories")

    # objects that belong to this category
    objects_fk: OneToMany = OneToManyField("Object", "category")

    def __str__(self):
        return self.name

    class Meta:
        pk = "name"
        ordering = ["name"]

    @cache
    def matching_synset(self, synset) -> bool:
        return synset.name in self.matching_synsets

    @cached_property
    def matching_synsets(self) -> Set["Synset"]:
        if not self.synset:
            return set()
        return {anc.name for anc in self.synset.ancestors} | {self.synset.name}

    @classmethod
    def view_error_mapped_to_non_leaf_synsets(cls):
        """Categories mapped to non-leaf synsets"""
        return [x for x in cls.all_objects() if len(x.synset.children) > 0]

    @classmethod
    def view_objectless(cls):
        """Categories that have no objects"""
        return [x for x in cls.all_objects() if len(x.objects) == 0]


@dataclass(eq=False, order=False)
class Object(Model):
    name: str
    # providing target
    provider: str = ""
    # bounding box size of the object in meters
    bounding_box_size: Optional[Tuple[float, float, float]] = None
    # the category that the object belongs to
    category_fk: ManyToOne = ManyToOneField(Category, "objects")
    # the particle system that the object belongs to
    particle_system_fk: ManyToOne = ManyToOneField(ParticleSystem, "particles")
    # the category of the object prior to getting renamed
    original_category_name: str = ""
    # meta links owned by the object
    meta_links_fk: ManyToMany = ManyToManyField(MetaLink, "on_objects")
    # roomobject counts of this object
    roomobjects_fk: OneToMany = OneToManyField("RoomObject", "object")
    # QA complaints for this object
    complaints_fk: OneToMany = OneToManyField("Complaint", "object")

    # attachment pairs this object participates in
    female_attachment_pairs_fk: ManyToMany = ManyToManyField(
        AttachmentPair, "female_objects"
    )
    male_attachment_pairs_fk: ManyToMany = ManyToManyField(
        AttachmentPair, "male_objects"
    )

    @property
    def owner(self):
        assert (self.category is None) != (
            self.particle_system is None
        ), f"{self.name} should either belong to a category or a particle system"
        if self.category is not None:
            return self.category
        elif self.particle_system is not None:
            return self.particle_system
        else:
            raise ValueError(
                f"Object {self.name} does not belong to a category or particle system"
            )

    def __str__(self):
        return self.owner.name + "-" + self.name

    class Meta:
        pk = "name"
        ordering = ["name"]

    @cache
    def matching_synset(self, synset) -> bool:
        return self.owner.matching_synset(synset)

    @cached_property
    def ready(self) -> bool:
        # An object is ready if it has no complaints.
        return len(list(self.complaints)) == 0

    @cached_property
    def state(self):
        if self.ready:
            return SynsetState.MATCHED
        else:
            return SynsetState.PLANNED

    @cached_property
    def image_url(self):
        model_id = self.name.split("-")[-1]
        return f"https://svl.stanford.edu/b1k/object_images/{model_id}_h264.mp4"

    def fully_supports_synset(self, synset, ignore=None) -> bool:
        all_required = synset.required_meta_links
        if ignore:
            all_required -= set(ignore)
        return all_required.issubset({x.name for x in self.meta_links})

    @cached_property
    def missing_meta_links(self) -> List[str]:
        if self.category is not None:
            return sorted(
                self.category.synset.required_meta_links
                - {x.name for x in self.meta_links}
            )
        elif self.particle_system is not None:
            return []

        raise ValueError(
            f"Object {self.name} does not belong to a category or particle system to check for missing meta links"
        )

    @classmethod
    def view_error_missing_meta_links(cls):
        """Objects with missing meta links (not including subpart)"""
        return [
            o
            for o in cls.all_objects()
            if o.category is not None
            and not o.fully_supports_synset(o.category.synset, ignore={"subpart"})
        ]


@dataclass(eq=False, order=False)
class Synset(Model):
    name: str
    # whether the synset is a custom synset or not
    is_custom: bool = field(default=False, repr=False)
    # wordnet definitions
    definition: str = field(default="", repr=False)
    # whether the synset is used as a substance in some task
    is_used_as_substance: bool = field(default=False, repr=False)
    # whether the synset is used as a non-substance in some task
    is_used_as_non_substance: bool = field(default=False, repr=False)
    # whether the synset is ever used as a fillable in any task
    is_used_as_fillable: bool = field(default=False, repr=False)
    # predicates the synset was used in as the first argument
    used_in_predicates_fk: ManyToMany = ManyToManyField(Predicate, "synsets")
    # all it's parents in the synset graph (NOTE: this does not include self)
    parents_fk: ManyToMany = ManyToManyField("Synset", "children")
    children_fk: ManyToMany = ManyToManyField("Synset", "parents")
    # all ancestors (NOTE: this does NOT include self)
    ancestors_fk: ManyToMany = ManyToManyField("Synset", "descendants")
    descendants_fk: ManyToMany = ManyToManyField("Synset", "ancestors")

    categories_fk: OneToMany = OneToManyField(Category, "synset")
    particle_systems_fk: OneToMany = OneToManyField(ParticleSystem, "synset")
    properties_fk: OneToMany = OneToManyField(Property, "synset")
    tasks_fk: ManyToMany = ManyToManyField("Task", "synsets")
    tasks_using_as_future_fk: ManyToMany = ManyToManyField("Task", "future_synsets")
    used_by_transition_rules_fk: ManyToMany = ManyToManyField(
        "TransitionRule", "input_synsets"
    )
    produced_by_transition_rules_fk: ManyToMany = ManyToManyField(
        "TransitionRule", "output_synsets"
    )
    machine_in_transition_rules_fk: ManyToMany = ManyToManyField(
        "TransitionRule", "machine_synsets"
    )
    roomsynsetrequirements_fk: OneToMany = OneToManyField(
        "RoomSynsetRequirement", "synset"
    )

    class Meta:
        pk = "name"
        ordering = ["name"]

    @cached_property
    def state(self) -> SynsetState:
        if self.name == "entity.n.01":
            return SynsetState.MATCHED  # root synset is always legal
        elif self.is_substance:
            if any(
                ps.state == SynsetState.MATCHED for ps in self.matching_particle_systems
            ):
                return SynsetState.MATCHED
            elif any(
                ps.state == SynsetState.PLANNED for ps in self.matching_particle_systems
            ):
                return SynsetState.PLANNED
            else:
                return SynsetState.UNMATCHED
        elif self.parents:
            if len(self.matching_ready_objects) > 0:
                return SynsetState.MATCHED
            elif len(self.matching_objects) > 0:
                return SynsetState.PLANNED
            else:
                return SynsetState.UNMATCHED
        else:
            return SynsetState.ILLEGAL

    @cached_property
    def property_names(self):
        return {prop.name for prop in self.properties}

    @cached_property
    def is_substance(self):
        return "substance" in self.property_names

    @cached_property
    def is_liquid(self):
        return "liquid" in self.property_names

    @cached_property
    def direct_matching_objects(self) -> Set[Object]:
        matched_objs = set()
        for category in self.categories:
            matched_objs.update(category.objects)
        return matched_objs

    @cached_property
    def direct_matching_ready_objects(self) -> Set[Object]:
        matched_objs = set()
        for category in self.categories:
            matched_objs.update(x for x in category.objects if x.ready)
        return matched_objs

    @cached_property
    def matching_objects(self) -> Set[Object]:
        matched_objs = set(self.direct_matching_objects)
        for synset in self.descendants:
            matched_objs.update(synset.direct_matching_objects)
        return matched_objs

    @cached_property
    def matching_ready_objects(self) -> Set[Object]:
        matched_objs = set(self.direct_matching_ready_objects)
        for synset in self.descendants:
            matched_objs.update(synset.direct_matching_ready_objects)
        return matched_objs

    @cached_property
    def matching_particle_systems(self) -> Set[ParticleSystem]:
        matched_particle_systems = set(self.particle_systems)
        for synset in self.descendants:
            matched_particle_systems.update(synset.particle_systems)
        return matched_particle_systems

    @cached_property
    def required_meta_links(self) -> Set[str]:
        properties = {
            prop.name: json.loads(prop.parameters) for prop in self.properties
        }
        return OBJECT_TAXONOMY.get_required_meta_links_for_abilities(properties)

    @cached_property
    def has_fully_supporting_object(self) -> bool:
        if self.is_liquid:
            return True

        for obj in self.matching_objects:
            if obj.fully_supports_synset(self):
                return True
        return False

    @cached_property
    def n_task_required(self):
        """Get whether the synset is required in any task, returns STATE METADATA"""
        return len(self.tasks)

    @cached_property
    def subgraph(self):
        """Get the edges of the subgraph of the synset"""
        G = nx.DiGraph()
        next_to_query = [(self, True, True)]
        while next_to_query:
            synset, query_parents, query_children = next_to_query.pop()
            if query_parents:
                for parent in synset.parents:
                    G.add_edge(parent, synset)
                    next_to_query.append((parent, True, False))
            if query_children:
                for child in synset.children:
                    G.add_edge(synset, child)
                    next_to_query.append((child, False, True))
        return G

    @cached_property
    def task_relevant(self):
        # Is it directly used in a task?
        if self.tasks:
            return True

        # Is it used in a transition that's used in a task?
        transition_relevant_synsets = {
            anc
            for t in Task.all_objects()
            for transition in t.relevant_transitions
            for s in list(transition.output_synsets) + list(transition.input_synsets)
            for anc in set(s.ancestors) | {s}
        }
        if self in transition_relevant_synsets:
            return True

        return False

    @cached_property
    def generates_synsets(self):
        # Look for particleApplier and particleSource annotations.
        generated_synsets = set()
        for prop in self.properties:
            if prop.name not in ["particleApplier", "particleSource"]:
                continue
            prop_params = json.loads(prop.parameters)
            conditions = prop_params.get("conditions", {})
            generated_synsets.update([Synset.get(name) for name in conditions.keys()])

        return generated_synsets

    @cached_property
    def relevant_transitions(self):
        return sorted(
            set(self.produced_by_transition_rules)
            | set(self.used_by_transition_rules)
            | set(self.machine_in_transition_rules)
        )

    @cached_property
    def transition_subgraph(self):
        G = nx.DiGraph()
        for transition in self.relevant_transitions:
            G.add_node(transition)
            for input_synset in transition.input_synsets:
                G.add_edge(input_synset, transition)
            for machine_synset in transition.machine_synsets:
                G.add_edge(machine_synset, transition)
            for output_synset in transition.output_synsets:
                G.add_edge(transition, output_synset)
        return nx.relabel_nodes(
            G,
            lambda x: (x.name if isinstance(x, Synset) else f"recipe: {x.name}"),
            copy=True,
        )

    def is_produceable_from(self, synsets):
        def _is_produceable_from(_self, _synsets, _seen):
            if _self in _seen:
                return False, set()

            # If it's already available, then we're good.
            if _self in _synsets:
                return True, set()

            # Otherwise, are there any recipes that I can use to obtain it?
            recipe_alternatives = set()
            for recipe in _self.produced_by_transition_rules:
                producabilities_and_recipe_sets = [
                    _is_produceable_from(ingredient, _synsets, _seen | {_self})
                    for ingredient in recipe.input_synsets
                ]
                producabilities, recipe_sets = zip(*producabilities_and_recipe_sets)
                if all(producabilities):
                    recipe_alternatives.add(recipe)
                    recipe_alternatives.update(
                        ingredient_recipe
                        for recipe_set in recipe_sets
                        for ingredient_recipe in recipe_set
                    )

            if not recipe_alternatives:
                return False, set()

            return True, recipe_alternatives

        return _is_produceable_from(self, synsets, set())

    @cached_property
    def is_derivative(self):
        derivative_words = ["cooked__", "half__", "diced__", "melted__"]
        return any(self.name.startswith(dw) for dw in derivative_words)

    @cached_property
    def derivative_parents(self):
        # Check if the synset is a derivative
        if not self.is_derivative:
            return None

        # Find the synset that has this as a child
        return {s for s in Synset.all_objects() if self in s.derivative_children}

    @cached_property
    def derivative_children_names(self):
        sliceable_children = []
        diceable_children = []
        cookable_children = []
        meltable_children = []

        for p in self.properties:
            if p.name == "sliceable":
                try:
                    sliceable_children.append(
                        json.loads(p.parameters)["sliceable_derivative_synset"]
                    )
                except KeyError:
                    raise ValueError(
                        f"'sliceable_derivative_synset' key not found in property parameters for {p.name} in {self.name}"
                    )
            elif p.name == "diceable":
                try:
                    diceable_children.append(
                        json.loads(p.parameters)["uncooked_diceable_derivative_synset"]
                    )
                except KeyError:
                    raise ValueError(
                        f"'uncooked_diceable_derivative_synset' key not found in property parameters for {p.name} in {self.name}"
                    )
            elif p.name == "cookable" and self.is_substance:
                try:
                    cookable_children.append(
                        json.loads(p.parameters)["substance_cooking_derivative_synset"]
                    )
                except KeyError:
                    raise ValueError(
                        f"'substance_cooking_derivative_synset' key not found in property parameters for {p.name} in {self.name}"
                    )
            elif p.name == "meltable":
                try:
                    meltable_children.append(
                        json.loads(p.parameters)["meltable_derivative_synset"]
                    )
                except KeyError:
                    raise ValueError(
                        f"'meltable_derivative_synset' key not found in property parameters for {p.name} in {self.name}"
                    )
        return set(
            sliceable_children
            + diceable_children
            + cookable_children
            + meltable_children
        )

    @cached_property
    def derivative_children(self):
        return {Synset.get(name) for name in self.derivative_children_names}

    @cached_property
    def derivative_ancestors(self):
        if not self.derivative_parents:
            return {self}
        return (
            {self}
            | self.derivative_parents
            | {
                ancestor
                for parent in self.derivative_parents
                for ancestor in parent.derivative_ancestors
            }
        )

    @cached_property
    def derivative_descendants(self):
        descendants = {self}
        descendants.update(self.derivative_children)
        for child in self.derivative_children:
            descendants.update(child.derivative_descendants)
        return descendants

    @classmethod
    def view_error_substance_mismatch(cls):
        """Synsets that are used in predicates that don't match their substance state"""
        return [
            s
            for s in cls.all_objects()
            if (
                (s.is_substance and s.is_used_as_non_substance)
                or (not s.is_substance and s.is_used_as_substance)
                or (s.is_used_as_substance and s.is_used_as_non_substance)
            )
        ]

    @classmethod
    def view_error_substance_missing_particle_system(cls):
        """Synsets that are marked as a substance but do not have a particle system"""
        return [
            s
            for s in cls.all_objects()
            if s.is_substance and len(s.matching_particle_systems) == 0
        ]

    @classmethod
    def view_error_substance_assigned_unrelated_category(cls):
        """Substance synsets that have assigned categories"""
        return [
            s for s in cls.all_objects() if s.is_substance and len(s.categories) > 0
        ]

    @classmethod
    def view_error_object_unsupported_properties(cls):
        """Synsets that do not have at least one object that supports all of annotated properties"""
        return [
            s
            for s in Synset.all_objects()
            if not s.is_substance and not s.has_fully_supporting_object
        ]

    @classmethod
    def view_error_unnecessary(cls):
        """Objectless synsets that are not used in any task or required by any property"""
        useful_synsets = set()

        # Iteratively search for useful synsets
        while True:
            task_relevant = {s for s in cls.all_objects() if s.task_relevant}
            has_objects = {s for s in cls.all_objects() if len(s.matching_objects) > 0}
            has_particle_system_with_particles = {
                s
                for s in cls.all_objects()
                if len(
                    [
                        particle
                        for ps in s.matching_particle_systems
                        for particle in ps.particles
                    ]
                )
                > 0
            }
            generated_by_useful = {
                generatee for s in useful_synsets for generatee in s.generates_synsets
            }
            ancestor_of_useful = {
                ancestor for s in useful_synsets for ancestor in s.ancestors
            }
            derivative_of_useful = {
                derivative
                for s in useful_synsets
                for derivative in s.derivative_ancestors | s.derivative_descendants
            }
            new_useful = (
                task_relevant
                | has_objects
                | has_particle_system_with_particles
                | generated_by_useful
                | ancestor_of_useful
                | derivative_of_useful
            )
            delta_useful = new_useful - useful_synsets
            if not delta_useful:
                break
            useful_synsets.update(delta_useful)

        return set(cls.all_objects()) - useful_synsets

    @classmethod
    def view_error_bad_derivative(cls):
        """Derivative synsets that exist even though the original synset is missing the expected property"""
        return [
            s for s in cls.all_objects() if s.is_derivative and not s.derivative_parents
        ]

    @classmethod
    def view_error_missing_derivative(cls):
        """Synsets that are missing a derivative synset that is expected to exist from property annotations"""
        return sorted(
            [
                s
                for s in cls.all_objects()
                if len(s.derivative_children_names) != len(s.derivative_children)
            ]
        )

    @classmethod
    def view_error_missing_definition(cls):
        """Synsets that are missing a definition"""
        return [s for s in cls.all_objects() if not s.definition]

    @classmethod
    def view_unmatched(self):
        """Synsets that are unmatched to either an object or a particle system"""
        return [s for s in self.all_objects() if s.state == SynsetState.UNMATCHED]


@dataclass(eq=False, order=False)
class TransitionRule(Model):
    name: str
    input_synsets_fk: ManyToMany = ManyToManyField(Synset, "used_by_transition_rules")
    output_synsets_fk: ManyToMany = ManyToManyField(
        Synset, "produced_by_transition_rules"
    )
    machine_synsets_fk: ManyToMany = ManyToManyField(
        Synset, "machine_in_transition_rules"
    )

    class Meta:
        pk = "name"
        ordering = ["name"]

    @cached_property
    def subgraph(self):
        G = nx.DiGraph()
        G.add_node(self)
        for input_synset in self.input_synsets:
            G.add_edge(input_synset, self)
        for machine_synset in self.machine_synsets:
            G.add_edge(machine_synset, self)
        for output_synset in self.output_synsets:
            G.add_edge(self, output_synset)
        return nx.relabel_nodes(
            G,
            lambda x: (x.name if isinstance(x, Synset) else f"recipe: {x.name}"),
            copy=True,
        )

    @staticmethod
    def get_graph():
        G = nx.DiGraph()
        for transition in TransitionRule.all_objects():
            G.add_node(transition)
            for input_synset in transition.input_synsets:
                G.add_edge(input_synset, transition)
            for machine_synset in transition.machine_synsets:
                G.add_edge(machine_synset, transition)
            for output_synset in transition.output_synsets:
                G.add_edge(transition, output_synset)
        return G


@dataclass(eq=False, order=False)
class Task(Model):
    name: str
    definition: str = field(default="", repr=False)
    synsets_fk: ManyToMany = ManyToManyField(
        Synset, "tasks"
    )  # the synsets required by this task
    future_synsets_fk: ManyToMany = ManyToManyField(
        Synset, "tasks_using_as_future"
    )  # the synsets that show up as future synsets in this task (e.g. don't exist in initial)
    uses_predicates_fk: ManyToMany = ManyToManyField(Predicate, "tasks")
    room_requirements_fk: OneToMany = OneToManyField("RoomRequirement", "task")

    class Meta:
        pk = "name"
        ordering = ["name"]

    def __str__(self):
        return self.name

    @cached_property
    def state(self):
        if (
            self.synset_state == SynsetState.MATCHED
            and self.scene_state == SynsetState.MATCHED
        ):
            return SynsetState.MATCHED
        elif (
            self.synset_state == SynsetState.UNMATCHED
            or self.scene_state == SynsetState.UNMATCHED
        ):
            return SynsetState.UNMATCHED
        else:
            return SynsetState.PLANNED

    def matching_scene(self, scene: Scene) -> str:
        """checks whether a scene satisfies task requirements"""
        ret = ""
        for room_requirement in self.room_requirements:
            scene_ret = f"Cannot find suitable {room_requirement.type}: "
            for room in scene.rooms:
                if room.type != room_requirement.type:
                    continue
                room_ret = room.matching_room_requirement(room_requirement)
                if len(room_ret) == 0:
                    scene_ret = ""
                    break
                else:
                    scene_ret += f"{room.name} is missing {room_ret}; "
            if len(scene_ret) > 0:
                ret += scene_ret[:-2] + "."
        return ret

    @cached_property
    def uses_transition(self):
        return len(self.relevant_transitions) > 0

    @cached_property
    def uses_visual_substance(self):
        return any(
            "visualSubstance" in synset.property_names for synset in self.synsets
        )

    @cached_property
    def uses_physical_substance(self):
        return any(
            "physicalSubstance" in synset.property_names for synset in self.synsets
        )

    @cached_property
    def uses_attachment(self):
        return any(pred.name == "attached" for pred in self.uses_predicates)

    @cached_property
    def uses_cloth(self):
        return any(
            pred.name in ["folded", "draped", "unfolded"]
            for pred in self.uses_predicates
        ) or any("cloth" in synset.property_names for synset in self.synsets)

    @cached_property
    def substance_synsets(self):
        """synsets that represent a substance"""
        return [x for x in self.synsets if x.is_substance]

    @cached_property
    def synset_state(self) -> str:
        if any(synset.state == SynsetState.ILLEGAL for synset in self.synsets):
            return SynsetState.UNMATCHED
        elif any(synset.state == SynsetState.UNMATCHED for synset in self.synsets):
            return SynsetState.UNMATCHED
        elif any(synset.state == SynsetState.PLANNED for synset in self.synsets):
            return SynsetState.PLANNED
        else:
            return SynsetState.MATCHED

    @cached_property
    def problem_synsets(self):
        return [
            synset
            for synset in self.synsets
            if synset.state in (SynsetState.ILLEGAL, SynsetState.UNMATCHED)
        ]

    @cached_property
    def scene_matching_dict(self) -> Dict[str, Dict[str, str]]:
        ret = {}
        for scene in Scene.all_objects():
            matching_result = self.matching_scene(scene=scene)
            ret[scene] = {
                "matched": len(matching_result) == 0,
                "reason": matching_result,
            }
        return ret

    @cached_property
    def matched_scenes(self) -> List[Scene]:
        """Get the list of scenes that match the task"""
        return [
            scene
            for scene, result in self.scene_matching_dict.items()
            if result["matched"]
        ]

    @cached_property
    def scene_state(self) -> str:
        scene_matching_dict = self.scene_matching_dict
        if any(x["matched"] for x in scene_matching_dict.values()):
            return SynsetState.MATCHED
        else:
            return SynsetState.UNMATCHED

    @cached_property
    def producability_data(self) -> Dict[Synset, Tuple[bool, Set[TransitionRule]]]:
        """Map each synset to a tuple of whether it is produceable and the transition rules that can be used to produce it"""
        starting_synsets = set(self.synsets) - set(self.future_synsets)
        return {
            synset: synset.is_produceable_from(starting_synsets)
            for synset in self.synsets
        }

    @cached_property
    def relevant_transitions(self):
        return [
            rule
            for synset, (producability, rules) in self.producability_data.items()
            for rule in rules
        ]

    @cached_property
    def transition_graph(self):
        G = nx.DiGraph()
        future_synsets = set(self.future_synsets)
        starting_synsets = set(self.synsets) - future_synsets

        def human_readable_name(s):
            if s in starting_synsets:
                return f"initial: {s.name}"
            elif s in future_synsets:
                return f"future: {s.name}"
            else:
                return s.name

        for synset in self.synsets:
            G.add_node(human_readable_name(synset), type="obj")
        for transition in self.relevant_transitions:
            transition_name = f"recipe: {transition.name}"
            G.add_node(transition_name, type="transition", text=transition.name)
            for input_synset in transition.input_synsets:
                G.add_edge(human_readable_name(input_synset), transition_name)
            for machine_synset in transition.machine_synsets:
                G.add_edge(human_readable_name(machine_synset), transition_name)
            for output_synset in transition.output_synsets:
                G.add_edge(transition_name, human_readable_name(output_synset))

        # Prune the graph so that it only contains all the future nodes and everything that can be used
        # to transition into them.
        future_nodes = {human_readable_name(s) for s in future_synsets}
        future_node_trees = [nx.bfs_tree(G.reverse(), node) for node in future_nodes]
        future_node_tree_nodes = [
            node for tree in future_node_trees for node in tree.nodes
        ]
        subgraph = G.subgraph(future_node_tree_nodes)

        return subgraph

    @cached_property
    def partial_transition_graph(self):
        future_synsets = set(self.future_synsets)
        starting_synsets = set(self.synsets) - future_synsets

        def human_readable_name(s):
            if isinstance(s, Synset):
                if s in starting_synsets:
                    return f"initial: {s.name}"
                elif s in future_synsets:
                    return f"future: {s.name}"
                else:
                    return f"missing: {s.name}"
            elif isinstance(s, TransitionRule):
                return f"recipe: {s.name}"
            else:
                raise ValueError("Unexpected node.")

        # Build a graph from all the transitions
        G = TransitionRule.get_graph()

        # Show all the nodes that show up in some path from the initial to the future synsets.
        all_nodes_on_path = set()
        for f, t in itertools.product(starting_synsets, future_synsets):
            if f not in G or t not in G:
                continue
            for p in nx.all_simple_paths(G, f, t):
                all_nodes_on_path.update(p)

        # Also add all ingredients of each recipe
        all_recipes = [x for x in all_nodes_on_path if isinstance(x, TransitionRule)]
        all_ingredients = {
            inp for recipe in all_recipes for inp in recipe.input_synsets
        }
        all_nodes_to_keep = all_nodes_on_path | all_ingredients

        # Get the subgraph, convert that to a text graph.
        subgraph = G.subgraph(all_nodes_to_keep)
        return nx.relabel_nodes(subgraph, human_readable_name, copy=True)

    @cached_property
    def unreachable_goal_synsets(self) -> List[str]:
        """Get a list of synsets that are in the goal but cannot be reached from the initial state"""
        return [s for s in self.future_synsets if not self.producability_data[s][0]]

    @cached_property
    def goal_is_reachable(self) -> bool:
        """Get whether the goal is reachable from the initial state"""
        return len(self.unreachable_goal_synsets) == 0

    @classmethod
    def view_error_transition_failure(cls):
        """Tasks that do not have a valid transition path from the initial to the goal"""
        return [x for x in cls.all_objects() if not x.goal_is_reachable]

    @classmethod
    def view_error_missing_scene(cls):
        """Tasks that are not matched to any scene"""
        return [x for x in cls.all_objects() if x.scene_state == SynsetState.UNMATCHED]

    @classmethod
    def view_error_missing_object(cls):
        """Tasks that are missing some objects or particle systems"""
        return [x for x in cls.all_objects() if x.synset_state == SynsetState.UNMATCHED]

    @classmethod
    def view_challenge(cls):
        """Tasks that are candidates for the challenge"""
        CHALLENGE_TASKS = [
            "putting_away_toys-0",
            "can_meat-0",
            "assembling_gift_baskets-0",
            "picking_up_trash-0",
            "putting_away_Halloween_decorations-0",
            "can_vegetables-0",
            "bringing_water-0",
            "cleaning_up_plates_and_food-0",
            "picking_up_toys-0",
            "clean_up_broken_glass-0",
            "carrying_in_groceries-0",
            "collecting_aluminum_cans-0",
            "clearing_the_table_after_dinner-0",
            "loading_the_dishwasher-0",
            "setting_mousetraps-0",
            "hiding_Easter_eggs-0",
            "sorting_mail-0",
            "make_dessert_watermelons-0",
            "rearranging_furniture-0",
            "rearranging_kitchen_furniture-0",
            "putting_up_Christmas_decorations_inside-0",
            "bringing_in_mail-0",
            "bringing_in_wood-0",
            "dispose_of_a_pizza_box-0",
            "opening_windows-0",
            "clean_up_your_desk-0",
            "preserving_meat-0",
            "bringing_glass_to_recycling-0",
            "watering_outdoor_flowers-0",
            "fill_a_bucket_in_a_small_sink-0",
            "can_syrup-0",
            "can_beans-0",
            "carrying_water-0",
            "filling_salt-0",
            "mixing_drinks-0",
            "adding_chemicals_to_pool-0",
            "spraying_for_bugs-0",
            "painting_porch-0",
            "installing_a_fence-0",
            "ice_cookies-0",
            "adding_chemicals_to_lawn-0",
            "watering_indoor_flowers-0",
            "cleaning_bathtub-0",
            "putting_up_shelves-0",
            "putting_on_license_plates-0",
            "installing_smoke_detectors-0",
            "hanging_pictures-0",
            "attach_a_camera_to_a_tripod-0",
            "setup_a_trampoline-0",
            "cook_hot_dogs-0",
            "make_a_steak-0",
            "toast_buns-0",
            "grill_burgers-0",
            "cook_bacon-0",
            "cook_a_brisket-0",
            "cool_cakes-0",
            "setting_the_fire-0",
            "freeze_pies-0",
            "freeze_fruit-0",
            "freeze_vegetables-0",
            "freeze_meat-0",
            "thawing_frozen_food-0",
            "thaw_frozen_fish-0",
            "heating_food_up-0",
            "reheat_frozen_or_chilled_food-0",
            "turning_on_radio-0",
            "mowing_the_lawn-0",
            "installing_a_modem-0",
            "installing_alarms-0",
            "turning_out_all_lights_before_sleep-0",
            "chop_an_onion-0",
            "slicing_vegetables-0",
            "chopping_wood-0",
            "make_spa_water-0",
            "cook_eggplant-0",
            "melt_white_chocolate-0",
            "cook_corn-0",
            "cook_a_crab-0",
            "cook_a_pumpkin-0",
            "boil_water-0",
            "boil_water_in_the_microwave-0",
            "toast_sunflower_seeds-0",
            "toast_coconut-0",
            "cook_chickpeas-0",
            "make_microwave_popcorn-0",
            "make_a_strawberry_slushie-0",
            "cook_soup-0",
            "make_a_milkshake-0",
            "make_chocolate_milk-0",
            "make_cake_mix-0",
            "make_cinnamon_sugar-0",
            "make_a_basic_brine-0",
            "make_a_cappuccino-0",
            "make_biscuits-0",
            "make_chocolate_biscuits-0",
            "make_cookies-0",
            "make_dinner_rolls-0",
            "clean_a_hamper-0",
            "clean_a_tie-0",
            "wash_a_baseball_cap-0",
        ]
        return [Task.get(x) for x in CHALLENGE_TASKS]


@dataclass(eq=False, order=False)
class RoomRequirement(Model):
    # TODO: make this one of the room types. enum?
    type: str
    id: str = UUIDField()
    task_fk: ManyToOne = ManyToOneField(Task, "room_requirements")
    roomsynsetrequirements_fk: OneToMany = OneToManyField(
        "RoomSynsetRequirement", "room_requirement"
    )

    class Meta:
        pk = "id"
        unique_together = ("task", "type")
        ordering = ["type"]


@dataclass(eq=False, order=False)
class RoomSynsetRequirement(Model):
    id: str = UUIDField()
    room_requirement_fk: ManyToOne = ManyToOneField(
        RoomRequirement, "roomsynsetrequirements"
    )
    synset_fk: ManyToOne = ManyToOneField(Synset, "roomsynsetrequirements")
    count: int = 0

    class Meta:
        pk = "id"
        unique_together = ("room_requirement", "synset")
        ordering = ["id"]  # TODO: synset__name when implemented


@dataclass(eq=False, order=False)
class Room(Model):
    name: str
    # type of the room
    # TODO: make this one of the room types
    type: str
    id: str = UUIDField()
    # the scene the room belongs to
    scene_fk: ManyToOne = ManyToOneField(Scene, "rooms")
    # the room objects in this object
    roomobjects_fk: OneToMany = OneToManyField("RoomObject", "room")

    class Meta:
        pk = "id"
        unique_together = ("name", "scene")
        ordering = ["name"]

    def __str__(self):
        return f"{self.scene.name}_{self.name}"

    @cached_property
    def non_clutter_roomobjects(self):
        return [x for x in self.roomobjects if not x.clutter]

    def matching_room_requirement(self, room_requirement: RoomRequirement) -> str:
        """
        checks whether the room satisfies the room requirement from a task
        returns an empty string if it does, otherwise returns a string describing what is missing
        """
        G = nx.Graph()
        # Add a node for each required object
        synset_node_to_synset = {}
        for room_synset_requirement in room_requirement.roomsynsetrequirements:
            synset_name = room_synset_requirement.synset.name
            for i in range(room_synset_requirement.count):
                node_name = f"{synset_name}_{i}"
                G.add_node(node_name)
                synset_node_to_synset[node_name] = room_synset_requirement.synset
        # Add a node for each object in the room
        for roomobject in self.non_clutter_roomobjects:
            for i in range(roomobject.count):
                object_name = f"{roomobject.object.name}_{i}"
                G.add_node(object_name)
                # Add edges to all matching synsets
                for synset_node, synset in synset_node_to_synset.items():
                    if roomobject.object.matching_synset(synset):
                        G.add_edge(object_name, synset_node)
        # Now do a bipartite matching
        M = nx.bipartite.maximum_matching(G, top_nodes=synset_node_to_synset.keys())
        # Now check that all required objects are matched
        if set(synset_node_to_synset.keys()).issubset(M.keys()):
            return ""
        else:
            missing_synsets = defaultdict(int)  # default value is 0
            for synset_node, synset in synset_node_to_synset.items():
                if synset_node not in M:
                    missing_synsets[synset.name] += 1
            return ", ".join(
                [f"{count} {synset}" for synset, count in missing_synsets.items()]
            )


@dataclass(eq=False, order=False)
class RoomObject(Model):
    id: str = UUIDField()
    # the room that the object belongs to
    room_fk: ManyToOne = ManyToOneField(Room, "roomobjects")
    # the actual object that the room object maps to
    object_fk: ManyToOne = ManyToOneField(Object, "roomobjects")
    # number of objects in the room
    count: int = 0
    # whether this count is for clutter objects or not
    clutter: bool = False

    class Meta:
        pk = "id"
        unique_together = ("room", "object")
        ordering = ["id"]  # TODO: ["room__name", "object__name"] when implemented

    def __str__(self):
        clutter_substr = "clutter" if self.clutter else "nonclutter"
        return f"{str(self.room)}_{self.object.name}_{clutter_substr}"


@dataclass(eq=False, order=False)
class ComplaintType(Model):
    name: str
    complaints_fk: OneToMany = OneToManyField("Complaint", "complaint_type")

    class Meta:
        pk = "name"
        ordering = ["name"]

    @cached_property
    def objects(self):
        return [complaint.object for complaint in self.complaints]


@dataclass(eq=False, order=False)
class Complaint(Model):
    id: str = UUIDField()
    object_fk: ManyToOne = ManyToOneField(Object, "complaints")
    complaint_type_fk: ManyToOne = ManyToOneField(ComplaintType, "complaints")
    prompt_additional_info: str = ""  # provided by the QA script as part of the prompt
    response: str = ""  # provided by the QAing user

    class Meta:
        pk = "id"
        ordering = ["id"]

    def __str__(self):
        return f"{self.object.name} - {self.complaint_type.name}: {self.response}"
