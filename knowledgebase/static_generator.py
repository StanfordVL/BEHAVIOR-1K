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
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from functools import partial
from jinja2 import Environment, FileSystemLoader, select_autoescape
from datetime import datetime

from bddl.knowledge_base import (
    Task, Scene, Synset, Category, Object, TransitionRule, 
    AttachmentPair, ComplaintType, ParticleSystem
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

class URLRegistry:
    """Registry for managing URLs in the static site."""
    
    def __init__(self):
        self.routes = {}
        self._register_routes()
    
    def _register_routes(self):
        """Register all known routes."""
        # Main routes
        self.routes['index'] = '/knowledgebase/index.html'
        
        # Store pk fields for each model
        self.pk_fields = {}
        
        # Model-based routes
        for model in MODELS:
            model_snake = snake_case(model.__name__)
            model_plural = pluralize(model_snake)
            pk_field = model.Meta.pk
            
            # Store pk field for this model
            self.pk_fields[model_snake] = pk_field
            
            # List view
            self.routes[f'{model_snake}_list'] = f'/knowledgebase/{model_plural}/index.html'
            
            # Detail view pattern (will be filled with actual PKs)
            self.routes[f'{model_snake}_detail'] = f'/knowledgebase/{model_plural}/{{{pk_field}}}.html'
        
        # Special routes
        self.routes['challenge_tasks'] = '/knowledgebase/challenge_tasks/index.html'
        self.routes['searchable_items_list'] = '/knowledgebase/searchable_items.json'
        self.routes['profile_badge_view'] = '/knowledgebase/profile/badge.svg'
        self.routes['profile_plot_view'] = '/knowledgebase/profile/plot.png'
    
    def url_for(self, endpoint: str, **kwargs) -> str:
        """Generate URL for a given endpoint."""
        if endpoint not in self.routes:
            # Check for custom view endpoints
            for model in MODELS:
                model_snake = snake_case(model.__name__)
                model_plural = pluralize(model_snake)
                if endpoint.endswith(f'_{model_plural}'):
                    # Custom view
                    view_name = endpoint.replace(f'_{model_plural}', '')
                    return f'/knowledgebase/{view_name}_{model_plural}/index.html'
            
            raise ValueError(f"Unknown endpoint: {endpoint}")
        
        url = self.routes[endpoint]
        
        # Replace placeholders with actual values
        for key, value in kwargs.items():
            url = url.replace(f'{{{key}}}', str(value))
        
        return url


class StaticSiteGenerator:
    """Generate static HTML site from Jinja2 templates."""
    
    def __init__(self, output_dir: str = 'build'):
        self.output_dir = Path(output_dir)
        self.url_registry = URLRegistry()
        self.env = self._setup_jinja_env()
        
        # Track all pages to generate
        self.pages_to_generate = []
        self._collect_pages()
    
    def _setup_jinja_env(self) -> Environment:
        """Setup Jinja2 environment with custom filters and functions."""
        template_dir = Path(__file__).parent / 'knowledgebase' / 'templates'
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Add custom filters
        env.filters['slugify'] = slugify_filter
        env.filters['status_color'] = status_color
        env.filters['status_color_transition_rule'] = status_color_transition_rule
        env.filters['tocolor'] = status_color  # Alias for tocolor filter
        
        # Add url_for function to global context
        env.globals['url_for'] = self.url_registry.url_for
        
        # Add SynsetState for templates
        from bddl.knowledge_base import SynsetState
        env.globals['SynsetState'] = SynsetState
        
        return env
    
    def _collect_pages(self):
        """Collect all pages that need to be generated."""
        # Index page
        self.pages_to_generate.append(('index', 'index.html', self._get_index_context()))
        
        # Generate pages for each model
        for model in MODELS:
            self._collect_model_pages(model)
        
        # Challenge tasks page
        self.pages_to_generate.append(
            ('challenge_tasks', 'task_list.html', self._get_challenge_tasks_context())
        )
        
        # Custom views (error views)
        self._collect_custom_views()
    
    def _collect_model_pages(self, model):
        """Collect all pages for a given model."""
        model_snake = snake_case(model.__name__)
        model_plural = pluralize(model_snake)
        
        # Get all objects
        objects = model.all_objects()
        
        # List page
        list_context = {
            f'{model_snake}_list': objects,
            'view': {'model': model}
        }
        self.pages_to_generate.append(
            (f'{model_snake}_list', f'{model_snake}_list.html', list_context)
        )
        
        # Detail pages
        pk = model.Meta.pk
        for obj in objects:
            detail_context = {
                model_snake: obj,
                'view': {'model': model}
            }
            pk_value = getattr(obj, pk)
            output_path = f'{model_plural}/{pk_value}.html'
            self.pages_to_generate.append(
                (f'{model_snake}_detail', f'{model_snake}_detail.html', detail_context, output_path)
            )
    
    def _collect_custom_views(self):
        """Collect custom view pages."""
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
                        f'{view_name}_{model_plural}': objects,
                        'view': {'model': model}
                    }
                    
                    # Determine template
                    template_name = f'{model_snake}_list.html'
                    output_path = f'{view_name}_{model_plural}/index.html'
                    
                    self.pages_to_generate.append(
                        (f'{view_name}_{model_plural}', template_name, context, output_path)
                    )
                    
                    # Track error views for index page
                    if name.startswith("view_error_"):
                        error_name = name.replace("view_error_", "").replace("_", " ").title()
                        error_views.append({
                            'title': f'{error_name} {model_plural}',
                            'endpoint': f'{view_name}_{model_plural}',
                            'count': len(objects)
                        })
        
        # Store error views for index context
        self.error_views = error_views
    
    def _get_index_context(self) -> dict:
        """Get context for index page."""
        from bddl.knowledge_base import SynsetState
        
        context = {
            'task_metadata': {
                'challenge_ready': sum([1 for task in Task.view_challenge() if task.synset_state == SynsetState.MATCHED and task.scene_state == SynsetState.MATCHED]),
                'total_ready': sum([1 for task in Task.all_objects() if task.synset_state == SynsetState.MATCHED and task.scene_state == SynsetState.MATCHED]),
                'challenge': len(list(Task.view_challenge())),
                'total': len(list(Task.all_objects())),
            },
            'synset_metadata': {
                'leaf': sum(1 for x in Synset.all_objects() if len(x.children) == 0),
                'total': sum(1 for x in Synset.all_objects()),
            },
            'object_metadata': {
                'ready': sum(1 for x in Object.all_objects() if x.state == SynsetState.MATCHED),
                'planned': sum(1 for x in Object.all_objects() if x.state == SynsetState.PLANNED),
                'total': sum(1 for x in Object.all_objects()),
                'categories': sum(1 for x in Category.all_objects()),
                'particle_systems': sum(1 for x in ParticleSystem.all_objects()),
            },
            'scene_metadata': {
                'challenge': len(list(Scene.view_challenge())),
                'total': len(list(Scene.all_objects()))
            },
            'error_views': self.error_views if hasattr(self, 'error_views') else [],
        }
        
        return context
    
    def _get_challenge_tasks_context(self) -> dict:
        """Get context for challenge tasks page."""
        return {
            'task_list': Task.view_challenge(),
            'view': {'model': Task}
        }
    
    def generate_page(self, page_info: tuple) -> Tuple[str, str]:
        """Generate a single page."""
        if len(page_info) == 3:
            endpoint, template_name, context = page_info
            output_path = self.url_registry.url_for(endpoint).replace('/knowledgebase/', '')
        else:
            endpoint, template_name, context, output_path = page_info
        
        # Render template
        template = self.env.get_template(template_name)
        html = template.render(**context)
        
        return output_path, html
    
    def generate_searchable_items(self) -> Tuple[str, str]:
        """Generate searchable items JSON."""
        items = []
        
        for model in MODELS:
            model_snake = snake_case(model.__name__)
            model_plural = pluralize(model_snake)
            pk = model.Meta.pk
            
            for obj in model.all_objects():
                pk_value = getattr(obj, pk)
                items.append({
                    'type': model.__name__,
                    'title': str(pk_value),
                    'url': self.url_registry.url_for(f'{model_snake}_detail', **{pk: pk_value})
                })
        
        return 'searchable_items.json', json.dumps(items, indent=2)
    
    def generate_profile_assets(self):
        """Generate profile badge and plot."""
        assets = []
        
        # Generate badge
        try:
            badge_svg = get_profile_badge_svg()
            if badge_svg:
                assets.append(('profile/badge.svg', badge_svg))
        except Exception as e:
            print(f"Warning: Could not generate profile badge: {e}")
        
        # Generate plot
        try:
            plot_png = get_profile_plot_png()
            if plot_png:
                assets.append(('profile/plot.png', plot_png))
        except Exception as e:
            print(f"Warning: Could not generate profile plot: {e}")
        
        return assets
    
    def write_file(self, rel_path: str, content: str | bytes):
        """Write content to file."""
        output_path = self.output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(content, str):
            output_path.write_text(content, encoding='utf-8')
        else:
            output_path.write_bytes(content)
        
        return str(output_path)
    
    def generate_site(self, max_workers: int = 1):
        """Generate the entire static site."""
        print(f"Generating static site with {len(self.pages_to_generate)} pages...")
        
        # Clear output directory
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        
        # Generate pages sequentially (no multiprocessing)
        generated_files = []
        
        for i, page_info in enumerate(self.pages_to_generate):
            try:
                rel_path, content = self.generate_page(page_info)
                file_path = self.write_file(rel_path, content)
                generated_files.append(file_path)
                
                # Progress indicator
                if (i + 1) % 50 == 0:
                    print(f"  Generated {i + 1} pages...")
                    
            except Exception as e:
                print(f"Error generating page {page_info[0]}: {e}")
        
        # Generate searchable items JSON
        rel_path, content = self.generate_searchable_items()
        file_path = self.write_file(rel_path, content)
        generated_files.append(file_path)
        
        # Generate profile assets
        for rel_path, content in self.generate_profile_assets():
            file_path = self.write_file(rel_path, content)
            generated_files.append(file_path)
        
        print(f"✓ Generated {len(generated_files)} files in {self.output_dir}")
        return generated_files


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
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate static site for BEHAVIOR-1K Knowledgebase")
    parser.add_argument('-o', '--output', default='build', help='Output directory (default: build)')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Number of parallel workers (default: 1)')
    
    args = parser.parse_args()
    
    generator = StaticSiteGenerator(output_dir=args.output)
    generator.generate_site(max_workers=args.workers)


if __name__ == '__main__':
    main()