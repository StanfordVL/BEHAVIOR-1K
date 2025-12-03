#!/usr/bin/env python3
"""
Static site generator for BEHAVIOR-1K Knowledgebase.
Generates static HTML files directly using Jinja2 without Flask dependency.
"""

import os
import json
import inspect
import unicodedata
import re
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from functools import partial
from jinja2 import Environment, FileSystemLoader, select_autoescape
from datetime import datetime
from tqdm import tqdm

from bddl.knowledge_base import (
    Task,
    Scene,
    Synset,
    Category,
    Object,
    TransitionRule,
    AttachmentPair,
    ComplaintType,
    ParticleSystem,
)

# Import necessary components from existing code
from knowledgebase.profile_utils import get_profile_badge_svg, get_profile_plot_png
from knowledgebase.filters import status_color, status_color_transition_rule

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

# Global variables for site generation
OUTPUT_DIR = Path("build")
ROUTES = {}
PK_FIELDS = {}
ERROR_VIEWS = []


def register_routes():
    """Register all known routes."""
    global ROUTES, PK_FIELDS

    # Main routes
    ROUTES["index"] = "/knowledgebase/index.html"

    # Model-based routes
    for model in MODELS:
        model_snake = snake_case(model.__name__)
        model_plural = pluralize(model_snake)
        pk_field = model.Meta.pk

        # Store pk field for this model
        PK_FIELDS[model_snake] = pk_field

        # List view
        ROUTES[f"{model_snake}_list"] = f"/knowledgebase/{model_plural}/index.html"

        # Detail view pattern (will be filled with actual PKs)
        ROUTES[f"{model_snake}_detail"] = (
            f"/knowledgebase/{model_plural}/{{{pk_field}}}.html"
        )

    # Special routes
    ROUTES["challenge_tasks"] = "/knowledgebase/challenge_tasks/index.html"
    ROUTES["searchable_items_list"] = "/knowledgebase/searchable_items.json"
    ROUTES["profile_badge_view"] = "/knowledgebase/profile/badge.svg"
    ROUTES["profile_plot_view"] = "/knowledgebase/profile/plot.png"


def url_for(endpoint: str, **kwargs) -> str:
    """Generate URL for a given endpoint."""
    if endpoint not in ROUTES:
        # Check for custom view endpoints
        for model in MODELS:
            model_snake = snake_case(model.__name__)
            model_plural = pluralize(model_snake)
            if endpoint.endswith(f"_{model_plural}"):
                # Custom view
                view_name = endpoint.replace(f"_{model_plural}", "")
                return f"/knowledgebase/{view_name}_{model_plural}/index.html"

        raise ValueError(f"Unknown endpoint: {endpoint}")

    url = ROUTES[endpoint]

    # Replace placeholders with actual values
    for key, value in kwargs.items():
        url = url.replace(f"{{{key}}}", str(value))

    return url


def setup_jinja_env() -> Environment:
    """Setup Jinja2 environment with custom filters and functions."""
    template_dir = Path(__file__).parent / "knowledgebase" / "templates"
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )

    # Add custom filters
    env.filters["slugify"] = slugify_filter
    env.filters["status_color"] = status_color
    env.filters["status_color_transition_rule"] = status_color_transition_rule
    env.filters["tocolor"] = status_color  # Alias for tocolor filter

    # Add url_for function to global context
    env.globals["url_for"] = url_for

    # Add SynsetState for templates
    from bddl.knowledge_base import SynsetState

    env.globals["SynsetState"] = SynsetState

    return env


def get_index_context() -> dict:
    """Get context for index page."""
    from bddl.knowledge_base import SynsetState

    context = {
        "task_metadata": {
            "challenge_ready": sum(
                [
                    1
                    for task in Task.view_challenge()
                    if task.synset_state == SynsetState.MATCHED
                    and task.scene_state == SynsetState.MATCHED
                ]
            ),
            "total_ready": sum(
                [
                    1
                    for task in Task.all_objects()
                    if task.synset_state == SynsetState.MATCHED
                    and task.scene_state == SynsetState.MATCHED
                ]
            ),
            "challenge": len(list(Task.view_challenge())),
            "total": len(list(Task.all_objects())),
        },
        "synset_metadata": {
            "leaf": sum(1 for x in Synset.all_objects() if len(x.children) == 0),
            "total": sum(1 for x in Synset.all_objects()),
        },
        "object_metadata": {
            "ready": sum(
                1 for x in Object.all_objects() if x.state == SynsetState.MATCHED
            ),
            "planned": sum(
                1 for x in Object.all_objects() if x.state == SynsetState.PLANNED
            ),
            "total": sum(1 for x in Object.all_objects()),
            "categories": sum(1 for x in Category.all_objects()),
            "particle_systems": sum(1 for x in ParticleSystem.all_objects()),
        },
        "scene_metadata": {
            "challenge": len(list(Scene.view_challenge())),
            "total": len(list(Scene.all_objects())),
        },
        "error_views": ERROR_VIEWS,
    }

    return context


def get_challenge_tasks_context() -> dict:
    """Get context for challenge tasks page."""
    return {"task_list": Task.view_challenge(), "view": {"model": Task}}


def collect_model_pages(model, pages_to_generate):
    """Collect all pages for a given model."""
    model_snake = snake_case(model.__name__)
    model_plural = pluralize(model_snake)

    # Get all objects
    objects = model.all_objects()

    # List page
    list_context = {f"{model_snake}_list": objects, "view": {"model": model}}
    pages_to_generate.append(
        (f"{model_snake}_list", f"{model_snake}_list.html", list_context)
    )

    # Detail pages
    pk = model.Meta.pk
    for obj in objects:
        detail_context = {model_snake: obj, "view": {"model": model}}
        pk_value = getattr(obj, pk)
        # Ensure output path includes knowledgebase/ prefix to match URL structure
        output_path = f"knowledgebase/{model_plural}/{pk_value}.html"
        pages_to_generate.append(
            (
                f"{model_snake}_detail",
                f"{model_snake}_detail.html",
                detail_context,
                output_path,
            )
        )


def collect_custom_views(pages_to_generate):
    """Collect custom view pages."""
    global ERROR_VIEWS
    error_views = []

    for model in MODELS:
        model_snake = snake_case(model.__name__)
        model_plural = pluralize(model_snake)

        # Find methods starting with 'view_'
        for name, method in inspect.getmembers(model):
            if name.startswith("view_"):
                view_name = name.replace("view_", "")

                # Get objects from the view method
                objects = method()

                # Create context
                context = {
                    f"{view_name}_{model_plural}": objects,
                    "view": {"model": model},
                }

                # Determine template
                template_name = f"{model_snake}_list.html"
                # Ensure output path includes knowledgebase/ prefix to match URL structure
                output_path = f"knowledgebase/{view_name}_{model_plural}/index.html"

                pages_to_generate.append(
                    (f"{view_name}_{model_plural}", template_name, context, output_path)
                )

                # Track error views for index page
                if name.startswith("view_error_"):
                    error_name = (
                        name.replace("view_error_", "").replace("_", " ").title()
                    )
                    error_views.append(
                        {
                            "title": f"{error_name} {model_plural}",
                            "endpoint": f"{view_name}_{model_plural}",
                            "count": len(objects),
                        }
                    )

    # Store error views for index context
    ERROR_VIEWS = error_views


def collect_pages():
    """Collect all pages that need to be generated."""
    pages_to_generate = []

    # Index page
    pages_to_generate.append(("index", "index.html", get_index_context()))

    # Generate pages for each model
    for model in MODELS:
        collect_model_pages(model, pages_to_generate)

    # Challenge tasks page
    pages_to_generate.append(
        ("challenge_tasks", "task_list.html", get_challenge_tasks_context())
    )

    # Custom views (error views)
    collect_custom_views(pages_to_generate)

    return pages_to_generate


def generate_page(page_info: tuple, env: Environment) -> Tuple[str, str]:
    """Generate a single page."""
    if len(page_info) == 3:
        endpoint, template_name, context = page_info
        # Convert URL path to file path, keeping the /knowledgebase/ prefix
        url_path = url_for(endpoint)
        # Keep the knowledgebase/ prefix in the file path
        output_path = url_path.replace("/knowledgebase/", "knowledgebase/")
    else:
        endpoint, template_name, context, output_path = page_info
        # Ensure output path is consistent with knowledgebase/ structure
        if not output_path.startswith("knowledgebase/"):
            output_path = f"knowledgebase/{output_path}"

    # Render template
    template = env.get_template(template_name)
    html = template.render(**context)

    return output_path, html


async def generate_page_async(page_info: tuple, env: Environment, pbar: tqdm) -> str:
    """Generate a single page asynchronously."""
    try:
        rel_path, content = generate_page(page_info, env)
        file_path = await write_file_async(rel_path, content)
        pbar.update(1)
        return file_path
    except Exception as e:
        print(f"Error generating page {page_info[0]}: {e}")
        pbar.update(1)
        return None


async def write_file_async(rel_path: str, content: str | bytes):
    """Write content to file asynchronously."""
    output_path = OUTPUT_DIR / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(content, str):
        output_path.write_text(content, encoding="utf-8")
    else:
        output_path.write_bytes(content)

    return str(output_path)


def generate_searchable_items() -> Tuple[str, str]:
    """Generate searchable items JSON."""
    items = []

    for model in MODELS:
        model_snake = snake_case(model.__name__)
        model_plural = pluralize(model_snake)
        pk = model.Meta.pk

        for obj in model.all_objects():
            pk_value = getattr(obj, pk)
            items.append(
                {
                    "type": model.__name__,
                    "title": str(pk_value),
                    "url": url_for(f"{model_snake}_detail", **{pk: pk_value}),
                }
            )

    return "knowledgebase/searchable_items.json", json.dumps(items, indent=2)


def generate_profile_assets():
    """Generate profile badge and plot."""
    assets = []

    # Generate badge
    try:
        badge_svg = get_profile_badge_svg()
        if badge_svg:
            assets.append(("knowledgebase/profile/badge.svg", badge_svg))
    except Exception as e:
        print(f"Warning: Could not generate profile badge: {e}")

    # Generate plot
    try:
        plot_png = get_profile_plot_png()
        if plot_png:
            assets.append(("knowledgebase/profile/plot.png", plot_png))
    except Exception as e:
        print(f"Warning: Could not generate profile plot: {e}")

    return assets


def write_file(rel_path: str, content: str | bytes):
    """Write content to file."""
    output_path = OUTPUT_DIR / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(content, str):
        output_path.write_text(content, encoding="utf-8")
    else:
        output_path.write_bytes(content)

    return str(output_path)


async def generate_site_async():
    """Generate the entire static site asynchronously."""
    # Setup
    register_routes()
    env = setup_jinja_env()
    pages_to_generate = collect_pages()

    print(f"Generating static site with {len(pages_to_generate)} pages...")

    # Clear output directory
    if OUTPUT_DIR.exists():
        import shutil

        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Generate pages concurrently with asyncio
    generated_files = []

    # Create progress bar
    pbar = tqdm(total=len(pages_to_generate), desc="Generating pages")

    # Create tasks for all pages
    tasks = [
        generate_page_async(page_info, env, pbar) for page_info in pages_to_generate
    ]

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None results (errors)
    generated_files = [result for result in results if result is not None]

    pbar.close()

    # Generate searchable items JSON
    rel_path, content = generate_searchable_items()
    file_path = write_file(rel_path, content)
    generated_files.append(file_path)

    # Generate profile assets
    for rel_path, content in generate_profile_assets():
        file_path = write_file(rel_path, content)
        generated_files.append(file_path)

    print(f"✓ Generated {len(generated_files)} files in {OUTPUT_DIR}")
    return generated_files


def generate_site():
    """Generate the entire static site."""
    return asyncio.run(generate_site_async())


# Utility functions
def pluralize(name: str) -> str:
    """Pluralize a model name."""
    return name + "s" if not name == "category" else "categories"


def snake_case(camel_case: str) -> str:
    """Convert CamelCase to snake_case."""
    return re.sub("(?<!^)(?=[A-Z])", "_", camel_case).lower()


def camel_case(snake_case: str) -> str:
    """Convert snake_case to CamelCase."""
    return "".join(word.title() for word in snake_case.split("_"))


def slugify_filter(value) -> str:
    """Slugify filter for templates."""
    value = str(value)
    value = (
        unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def main():
    """Main entry point."""
    generate_site()


if __name__ == "__main__":
    main()
