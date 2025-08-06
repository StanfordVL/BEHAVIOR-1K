import inspect
import io
import re
from flask.views import View
from flask import render_template, send_file, url_for, jsonify
from bddl.knowledge_base import Task, Scene, Synset, Category, Object, TransitionRule, AttachmentPair, SynsetState, ParticleSystem

from . import profile_utils


def camel_to_snake(name: str) -> str:
    """Converts a CamelCase string to snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class TemplateView(View):
    def get_context_data(self):
        return {"view": self, "SynsetState": SynsetState}

    def get_template_name(self):
        return self.template_name

    def dispatch_request(self, **kwargs):
        return render_template(self.get_template_name(), **self.get_context_data(**kwargs))


class ListView(TemplateView):
    def get_template_name(self):
        try:
            return super().get_template_name()
        except AttributeError:
            return f"{camel_to_snake(self.model.__name__)}_list.html"

    def get_queryset(self):
        return list(self.model.all_objects())

    def get_context_data(self):
        context = super().get_context_data()
        context[self.context_object_name] = self.get_queryset()
        return context


class DetailView(TemplateView):
    model = Category
    context_object_name = "category"
    slug_field = "name"
    slug_url_kwarg = "category_name"

    def get_template_name(self):
        try:
            super().get_template_name()
        except AttributeError:
            return f"{camel_to_snake(self.model.__name__)}_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data()
        assert len(kwargs) == 1 and self.slug_url_kwarg in kwargs
        lookup_kwargs = {self.slug_field: kwargs[self.slug_url_kwarg]}
        context[self.context_object_name] = self.model.get(**lookup_kwargs)
        return context


class IndexView(TemplateView):
    template_name = "index.html"

    def __init__(self, error_views, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_views = error_views

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # task metadata
        context["task_metadata"] = {
            "challenge_ready": sum([1 for task in Task.view_challenge() if task.synset_state == SynsetState.MATCHED and task.scene_state == SynsetState.MATCHED]),
            "total_ready": sum([1 for task in Task.all_objects() if task.synset_state == SynsetState.MATCHED and task.scene_state == SynsetState.MATCHED]),
            "challenge": len(list(Task.view_challenge())),
            "total": len(list(Task.all_objects())),
        }
        # synset metadata
        context["synset_metadata"] = {
            "leaf": sum(1 for x in Synset.all_objects() if len(x.children) == 0),
            "total": sum(1 for x in Synset.all_objects()),
        }
        # object metadata
        context["object_metadata"] = {
            "ready": sum(1 for x in Object.all_objects() if x.state == SynsetState.MATCHED),
            "planned": sum(1 for x in Object.all_objects() if x.state == SynsetState.PLANNED),
            "total": sum(1 for x in Object.all_objects()),
            "categories": sum(1 for x in Category.all_objects()),
            "particle_systems": sum(1 for x in ParticleSystem.all_objects()),
        }
        # scene metadata
        context["scene_metadata"] = {
            "challenge": len(list(Scene.view_challenge())),
            "total": len(list(Scene.all_objects()))
        }
        context["error_views"] = [
            (view_name, view_class.page_title, len(view_class().get_queryset()))
            for view_name, view_class in self.error_views
        ]
        return context
    

def searchable_items_list():
    SEARCHABLE_ITEM_TYPES = [
        (Object, "Object", "object_detail"),
        (Category, "Category", "category_detail"), 
        (ParticleSystem, "Particle System", "particle_system_detail"), 
        (Synset, "Synset", "synset_detail"),
        (Task, "Task", "task_detail"),
        (Scene, "Scene", "scene_detail"),
        (TransitionRule, "Transition Rule", "transition_rule_detail"),
        (AttachmentPair, "Attachment Pair", "attachment_pair_detail")
    ]
    searchable_items = []
    for item_type, type_str, detail_view_name in SEARCHABLE_ITEM_TYPES:
        for item in item_type.all_objects():
            item_title = item.name if item_type != Object else str(item)
            searchable_items.append({"type": type_str, "title": item_title, "url": url_for(detail_view_name, name=item.name)})
    return jsonify(searchable_items)


def profile_plot_view():
    plot_img = profile_utils.plot_profile("Realtime Performance", 10, ignore_series=["Empty scene"])
    stream = io.BytesIO()
    plot_img.save(stream, format="png")
    stream.seek(0)
    return send_file(stream, mimetype="image/png")


def profile_badge_view():
    badge_text = profile_utils.make_realtime_badge("Rs_int")
    stream = io.BytesIO()
    stream.write(badge_text.encode("utf-8"))
    stream.seek(0)
    return send_file(stream, mimetype='image/svg+xml')
