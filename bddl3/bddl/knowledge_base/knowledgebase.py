from typing import Any, Dict, List, Optional


class KnowledgeBase:
    """Container for one fully materialized BDDL knowledge base instance."""

    def __init__(self, populate: bool = False, verbose: bool = True):
        self.properties: List[Any] = []
        self.properties_by_id: Dict[str, Any] = {}
        self.meta_links: List[Any] = []
        self.meta_links_by_name: Dict[str, Any] = {}
        self.attachment_pairs: List[Any] = []
        self.attachment_pairs_by_name: Dict[str, Any] = {}
        self.predicates: List[Any] = []
        self.predicates_by_name: Dict[str, Any] = {}
        self.scenes: List[Any] = []
        self.scenes_by_name: Dict[str, Any] = {}
        self.particle_systems: List[Any] = []
        self.particle_systems_by_name: Dict[str, Any] = {}
        self.categories: List[Any] = []
        self.categories_by_name: Dict[str, Any] = {}
        self.objects: List[Any] = []
        self.objects_by_name: Dict[str, Any] = {}
        self.synsets: List[Any] = []
        self.synsets_by_name: Dict[str, Any] = {}
        self.transition_rules: List[Any] = []
        self.transition_rules_by_name: Dict[str, Any] = {}
        self.tasks: List[Any] = []
        self.tasks_by_name: Dict[str, Any] = {}
        self.room_requirements: List[Any] = []
        self.room_requirements_by_id: Dict[str, Any] = {}
        self.roomsynsetrequirements: List[Any] = []
        self.roomsynsetrequirements_by_id: Dict[str, Any] = {}
        self.rooms: List[Any] = []
        self.rooms_by_id: Dict[str, Any] = {}
        self.roomobjects: List[Any] = []
        self.roomobjects_by_id: Dict[str, Any] = {}
        self.complaint_types: List[Any] = []
        self.complaint_types_by_name: Dict[str, Any] = {}
        self.complaints: List[Any] = []
        self.complaints_by_id: Dict[str, Any] = {}

        if populate:
            from bddl.knowledge_base.processing import populate_knowledgebase

            populate_knowledgebase(self, verbose=verbose)

    def add_synset(self, name: str, definition: str = "", is_custom: bool = False):
        from bddl.knowledge_base.models import Synset
        if name in self.synsets_by_name:
            raise ValueError(f"Duplicate key {name} for Synset")
        obj = Synset(name=name, definition=definition, is_custom=is_custom)
        self.synsets.append(obj)
        self.synsets_by_name[name] = obj
        obj.knowledgebase = self
        return obj

    def get_synset(self, name: str): return self.synsets_by_name.get(name)
    def all_synsets(self): return list(self.synsets)

    def add_category(self, name: str, synset=None):
        from bddl.knowledge_base.models import Category
        if name in self.categories_by_name:
            raise ValueError(f"Duplicate key {name} for Category")
        obj = Category(name=name, synset=synset)
        self.categories.append(obj)
        self.categories_by_name[name] = obj
        obj.knowledgebase = self
        return obj

    def get_category(self, name: str): return self.categories_by_name.get(name)
    def all_categories(self): return list(self.categories)

    def add_particle_system(self, name: str, synset=None, parameters: Optional[str] = None):
        from bddl.knowledge_base.models import ParticleSystem
        if name in self.particle_systems_by_name:
            raise ValueError(f"Duplicate key {name} for ParticleSystem")
        obj = ParticleSystem(name=name, synset=synset, parameters=parameters)
        self.particle_systems.append(obj)
        self.particle_systems_by_name[name] = obj
        obj.knowledgebase = self
        return obj

    def get_particle_system(self, name: str): return self.particle_systems_by_name.get(name)
    def all_particle_systems(self): return list(self.particle_systems)

    def add_property(self, name: str, parameters: str, synset=None):
        from bddl.knowledge_base.models import Property
        obj = Property(name=name, parameters=parameters, synset=synset)
        if obj.id in self.properties_by_id:
            raise ValueError(f"Duplicate key {obj.id} for Property")
        self.properties.append(obj)
        self.properties_by_id[obj.id] = obj
        obj.knowledgebase = self
        return obj

    def all_properties(self): return list(self.properties)

    def add_object(self, name: str, provider: str = "", original_category_name: str = ""):
        from bddl.knowledge_base.models import Object
        if name in self.objects_by_name:
            raise ValueError(f"Duplicate key {name} for Object")
        obj = Object(name=name, provider=provider, original_category_name=original_category_name)
        self.objects.append(obj)
        self.objects_by_name[name] = obj
        obj.knowledgebase = self
        return obj

    def get_object(self, name: str): return self.objects_by_name.get(name)
    def all_objects(self): return list(self.objects)

    def add_scene(self, name: str):
        from bddl.knowledge_base.models import Scene
        if name in self.scenes_by_name:
            raise ValueError(f"Duplicate key {name} for Scene")
        obj = Scene(name=name)
        self.scenes.append(obj)
        self.scenes_by_name[name] = obj
        obj.knowledgebase = self
        return obj

    def get_scene(self, name: str): return self.scenes_by_name.get(name)
    def all_scenes(self): return list(self.scenes)

    def add_room(self, name: str, type: str, scene=None):
        from bddl.knowledge_base.models import Room
        obj = Room(name=name, type=type, scene=scene)
        if obj.id in self.rooms_by_id:
            raise ValueError(f"Duplicate key {obj.id} for Room")
        self.rooms.append(obj)
        self.rooms_by_id[obj.id] = obj
        obj.knowledgebase = self
        return obj

    def all_rooms(self): return list(self.rooms)

    def add_room_object(self, room=None, object=None, count: int = 0, clutter: bool = False):
        from bddl.knowledge_base.models import RoomObject
        obj = RoomObject(room=room, object=object, count=count, clutter=clutter)
        if obj.id in self.roomobjects_by_id:
            raise ValueError(f"Duplicate key {obj.id} for RoomObject")
        self.roomobjects.append(obj)
        self.roomobjects_by_id[obj.id] = obj
        obj.knowledgebase = self
        return obj

    def add_predicate(self, name: str):
        from bddl.knowledge_base.models import Predicate
        if name in self.predicates_by_name:
            raise ValueError(f"Duplicate key {name} for Predicate")
        obj = Predicate(name=name)
        self.predicates.append(obj)
        self.predicates_by_name[name] = obj
        obj.knowledgebase = self
        return obj

    def get_predicate(self, name: str): return self.predicates_by_name.get(name)
    def all_predicates(self): return list(self.predicates)

    def add_task(self, name: str, definition: str = ""):
        from bddl.knowledge_base.models import Task
        if name in self.tasks_by_name:
            raise ValueError(f"Duplicate key {name} for Task")
        obj = Task(name=name, definition=definition)
        self.tasks.append(obj)
        self.tasks_by_name[name] = obj
        obj.knowledgebase = self
        return obj

    def get_task(self, name: str): return self.tasks_by_name.get(name)
    def all_tasks(self): return list(self.tasks)

    def add_transition_rule(self, name: str):
        from bddl.knowledge_base.models import TransitionRule
        if name in self.transition_rules_by_name:
            raise ValueError(f"Duplicate key {name} for TransitionRule")
        obj = TransitionRule(name=name)
        self.transition_rules.append(obj)
        self.transition_rules_by_name[name] = obj
        obj.knowledgebase = self
        return obj

    def all_transition_rules(self): return list(self.transition_rules)

    def add_meta_link(self, name: str):
        from bddl.knowledge_base.models import MetaLink
        if name in self.meta_links_by_name:
            raise ValueError(f"Duplicate key {name} for MetaLink")
        obj = MetaLink(name=name)
        self.meta_links.append(obj)
        self.meta_links_by_name[name] = obj
        obj.knowledgebase = self
        return obj

    def get_meta_link(self, name: str): return self.meta_links_by_name.get(name)

    def add_attachment_pair(self, name: str):
        from bddl.knowledge_base.models import AttachmentPair
        if name in self.attachment_pairs_by_name:
            raise ValueError(f"Duplicate key {name} for AttachmentPair")
        obj = AttachmentPair(name=name)
        self.attachment_pairs.append(obj)
        self.attachment_pairs_by_name[name] = obj
        obj.knowledgebase = self
        return obj

    def get_attachment_pair(self, name: str): return self.attachment_pairs_by_name.get(name)
    def all_attachment_pairs(self): return list(self.attachment_pairs)

    def add_room_requirement(self, type: str, task=None):
        from bddl.knowledge_base.models import RoomRequirement
        obj = RoomRequirement(type=type, task=task)
        if obj.id in self.room_requirements_by_id:
            raise ValueError(f"Duplicate key {obj.id} for RoomRequirement")
        self.room_requirements.append(obj)
        self.room_requirements_by_id[obj.id] = obj
        obj.knowledgebase = self
        return obj

    def add_roomsynset_requirement(self, room_requirement=None, synset=None, count: int = 0):
        from bddl.knowledge_base.models import RoomSynsetRequirement
        obj = RoomSynsetRequirement(room_requirement=room_requirement, synset=synset, count=count)
        if obj.id in self.roomsynsetrequirements_by_id:
            raise ValueError(f"Duplicate key {obj.id} for RoomSynsetRequirement")
        self.roomsynsetrequirements.append(obj)
        self.roomsynsetrequirements_by_id[obj.id] = obj
        obj.knowledgebase = self
        return obj

    def add_complaint_type(self, name: str):
        from bddl.knowledge_base.models import ComplaintType
        if name in self.complaint_types_by_name:
            raise ValueError(f"Duplicate key {name} for ComplaintType")
        obj = ComplaintType(name=name)
        self.complaint_types.append(obj)
        self.complaint_types_by_name[name] = obj
        obj.knowledgebase = self
        return obj

    def get_complaint_type(self, name: str): return self.complaint_types_by_name.get(name)
    def all_complaint_types(self): return list(self.complaint_types)

    def add_complaint(self, object=None, complaint_type=None, prompt_additional_info: str = "", response: str = ""):
        from bddl.knowledge_base.models import Complaint
        obj = Complaint(object=object, complaint_type=complaint_type, prompt_additional_info=prompt_additional_info, response=response)
        if obj.id in self.complaints_by_id:
            raise ValueError(f"Duplicate key {obj.id} for Complaint")
        self.complaints.append(obj)
        self.complaints_by_id[obj.id] = obj
        obj.knowledgebase = self
        return obj

    def sort_all(self):
        self.properties.sort(key=lambda x: x.name)
        self.meta_links.sort(key=lambda x: x.name)
        self.attachment_pairs.sort(key=lambda x: x.name)
        self.predicates.sort(key=lambda x: x.name)
        self.scenes.sort(key=lambda x: x.name)
        self.particle_systems.sort(key=lambda x: x.name)
        self.categories.sort(key=lambda x: x.name)
        self.objects.sort(key=lambda x: x.name)
        self.synsets.sort(key=lambda x: x.name)
        self.transition_rules.sort(key=lambda x: x.name)
        self.tasks.sort(key=lambda x: x.name)
        self.room_requirements.sort(key=lambda x: x.type)
        self.roomsynsetrequirements.sort(key=lambda x: x.id)
        self.rooms.sort(key=lambda x: x.name)
        self.roomobjects.sort(key=lambda x: x.id)
        self.complaint_types.sort(key=lambda x: x.name)
        self.complaints.sort(key=lambda x: x.id)
