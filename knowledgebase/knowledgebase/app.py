import inspect
import unicodedata, re
from flask import redirect
from . import create_app
from bddl.knowledge_base import Task, Scene, Synset, Category, Object, TransitionRule, AttachmentPair, ComplaintType, ParticleSystem
from knowledgebase.views import ListView, DetailView, IndexView, profile_badge_view, profile_plot_view, searchable_items_list

MODELS = [
  AttachmentPair,
  Category,
  ComplaintType,
  Object,
  ParticleSystem,
  Scene,
  Synset,
  Task,
  TransitionRule,
]

def pluralize(name):
  return name + "s" if not name == "category" else "categories"


def snake_case(camel_case):
  return re.sub("(?<!^)(?=[A-Z])", "_", camel_case).lower()


def camel_case(snake_case):
  return "".join(word.title() for word in snake_case.split("_"))


app = create_app()
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True  # Add this line to enable Jinja2 auto-reload
app.debug = True  # This enables auto-reload for Python files

@app.template_filter('slugify')
def slugify_filter(value):
  value = str(value)
  value = (
    unicodedata.normalize("NFKD", value)
    .encode("ascii", "ignore")
    .decode("ascii")
  )
  value = re.sub(r"[^\w\s-]", "", value.lower())
  return re.sub(r"[-\s]+", "-", value).strip("-_")

@app.route("/", methods=["GET"])
def redirect_index():
  return redirect("/knowledgebase", code=302)

def add_url_rule(rule, view_func):
  print(f"Adding URL rule: {rule} -> {view_func.__name__}")
  app.add_url_rule(rule, view_func=view_func)

# Create the list, detail, error, and custom views for each model.
error_views = []  # (name, class)
for model in MODELS:
  # First, get relevant metadata from the model
  pk = model.Meta.pk
  model_snake_case = snake_case(model.__name__)
  detail_name = model_snake_case + "_detail"
  list_name = model_snake_case + "_list"
  plural_for_url = pluralize(model_snake_case)

  # Create the list view
  class_list_view = type(
    f"{model}ListView",
    (ListView,),
    {
      "model": model,
      "context_object_name": list_name,
    }
  )
  add_url_rule(f"/knowledgebase/{plural_for_url}/", view_func=class_list_view.as_view(list_name))

  # Create the detail view
  class_detail_view = type(
    f"{model}DetailView",
    (DetailView,),
    {
      "model": model,
      "context_object_name": model_snake_case,
      "slug_field": model.Meta.pk,
      "slug_url_kwarg": model.Meta.pk,
    }
  )
  add_url_rule(
    f"/knowledgebase/{plural_for_url}/<{pk}>/",
    view_func=class_detail_view.as_view(detail_name)
  )

  # Now walk through the view functions that the model provides, and add them
  # to the app. These are custom views - the error views are shown on the index, other
  # custom views need to be added to the navbar manually.
  # inspect the class to find all static methods that start with "view_"
  for func_name, func in inspect.getmembers(model):
    # Check if it's a view, if it's an error view, and get the view name
    if not func_name.startswith("view_"):
      continue
    custom_view_name = func_name[5:]  # Remove the "view_" prefix
    is_error = custom_view_name.startswith("error_")
    if is_error:
      custom_view_name = custom_view_name[6:]  # Remove the "error_" prefix

    # Suffix the view name with the pluralized model name
    custom_view_name = f"{custom_view_name}_{plural_for_url}"

    # Get a camel case view name
    view_class_name = camel_case(custom_view_name) + "View"

    # Create the queryset getter using the return value of the view function
    get_queryset = lambda self, func_name=func_name: getattr(self.model, func_name)()

    # Use inspect to get the docstring of the function
    page_title = inspect.getdoc(func)

    # Create a new view class
    custom_class_view = type(
      view_class_name,
      (class_list_view,),
      {
        "page_title": page_title,
        "get_queryset": get_queryset,
      }
    )
    # Add the view to the app
    add_url_rule(
      f"/knowledgebase/{custom_view_name}/",
      view_func=custom_class_view.as_view(custom_view_name)
    )
    if is_error:
      # If the view is an error view, add it to the error_url_patterns
      error_views.append((custom_view_name, custom_class_view))

add_url_rule("/knowledgebase/", view_func=IndexView.as_view("index", error_views=error_views))

# Add the profile views
add_url_rule("/knowledgebase/profile/badge.svg", view_func=profile_badge_view)
add_url_rule("/knowledgebase/profile/plot.png", view_func=profile_plot_view)
add_url_rule("/knowledgebase/searchable_items.json", view_func=searchable_items_list)
