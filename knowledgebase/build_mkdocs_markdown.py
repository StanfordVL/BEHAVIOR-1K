#!/usr/bin/env python3
"""
Markdown generator for BEHAVIOR-1K Knowledgebase.
Generates Markdown files for integration with MkDocs.
"""

import json
import inspect
import re
from pathlib import Path
import argparse

from bddl.knowledge_base import (
    Task, Scene, Synset, Category, Object, TransitionRule, 
    AttachmentPair, ComplaintType, ParticleSystem, SynsetState
)

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

class MarkdownGenerator:
    """Generates Markdown files for the knowledgebase."""
    
    def __init__(self, output_dir: Path = Path("docs/knowledgebase")):
        self.output_dir = output_dir
        self.generated_files = []
        
    def setup(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each model type
        for model in MODELS:
            subdir = self.output_dir / self._pluralize(self._snake_case(model.__name__))
            subdir.mkdir(exist_ok=True)
            
    def generate_all(self):
        """Generate all Markdown files."""
        print("Setting up directories...")
        self.setup()
        
        print("Generating index page...")
        self.generate_index()
        
        print("Generating model pages...")
        for model in MODELS:
            print(f"  Processing {model.__name__}...")
            self.generate_model_pages(model)
        
        print("Generating navigation file...")
        self.generate_nav_file()
        
        return self.generated_files
    
    def generate_index(self):
        """Generate the main index page."""
        # Gather statistics
        stats = {
            'task_metadata': {
                'challenge_ready': sum([1 for task in Task.view_challenge() 
                                      if task.synset_state == SynsetState.MATCHED 
                                      and task.scene_state == SynsetState.MATCHED]),
                'total_ready': sum([1 for task in Task.all_objects() 
                                  if task.synset_state == SynsetState.MATCHED 
                                  and task.scene_state == SynsetState.MATCHED]),
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
            }
        }
        
        # Collect error views
        error_views = []
        for model in MODELS:
            model_snake = self._snake_case(model.__name__)
            model_plural = self._pluralize(model_snake)
            
            for name, method in inspect.getmembers(model):
                if name.startswith("view_error_"):
                    error_name = name.replace("view_error_", "").replace("_", " ").title()
                    objects = method()
                    count = len(list(objects))
                    error_views.append({
                        'title': f'{error_name} {model_plural}',
                        'url': f'{name.replace("view_", "")}_{model_plural}/index.md',
                        'count': count
                    })
        
        content = f"""# BEHAVIOR-1K Knowledgebase Dashboard

<div class="grid cards" markdown>

-   :material-clipboard-list:{{ .lg .middle }} **[Tasks](tasks/index.md)**

    ---

    **Challenge Tasks**  
    :material-check-circle:{{ .mdx-pulse style="color: var(--md-primary-fg-color)" }} Sampleable: **{stats['task_metadata']['challenge_ready']}** / Total: **{stats['task_metadata']['challenge']}**

    **All Tasks**  
    :material-check-circle:{{ .mdx-pulse style="color: var(--md-primary-fg-color)" }} Sampleable: **{stats['task_metadata']['total_ready']}** / Total: **{stats['task_metadata']['total']}**

-   :material-cube-outline:{{ .lg .middle }} **[Objects](objects/index.md)**

    ---

    **Status**  
    :material-check:{{ style="color: green" }} Ready: **{stats['object_metadata']['ready']}** | :material-clock:{{ style="color: orange" }} In Progress: **{stats['object_metadata']['planned']}** | Total: **{stats['object_metadata']['total']}**

    **Organization**  
    Categories: **{stats['object_metadata']['categories']}** | Particle Systems: **{stats['object_metadata']['particle_systems']}**

-   :material-tag-multiple:{{ .lg .middle }} **[Synsets](synsets/index.md)**

    ---

    **Hierarchy**  
    :material-leaf:{{ style="color: green" }} Leaf Synsets: **{stats['synset_metadata']['leaf']}** / Total: **{stats['synset_metadata']['total']}**

-   :material-home-city:{{ .lg .middle }} **[Scenes](scenes/index.md)**

    ---

    **Available**  
    :material-flag:{{ style="color: var(--md-primary-fg-color)" }} Challenge Scenes: **{stats['scene_metadata']['challenge']}** / Total: **{stats['scene_metadata']['total']}**

</div>"""
        
        # Add errors section if there are any
        if error_views:
            content += """

## Issues

| Issue Type | Status |
|------------|--------|
"""
            for error in error_views:
                if error['count'] == 0:
                    status = ":material-check-circle:{{ style=\"color: green\" }} resolved"
                else:
                    status = f":material-alert-circle:{{ style=\"color: red\" }} **{error['count']}** issues"
                content += f"| {error['title']} | {status} |\n"
        
        output_path = self.output_dir / "index.md"
        output_path.write_text(content)
        self.generated_files.append(str(output_path))
        
    def generate_model_pages(self, model):
        """Generate list and detail pages for a model."""
        model_snake = self._snake_case(model.__name__)
        model_plural = self._pluralize(model_snake)
        
        # Generate list page
        self.generate_list_page(model, model_snake, model_plural)
        
        # Generate detail pages
        self.generate_detail_pages(model, model_snake, model_plural)
        
    def generate_list_page(self, model, model_snake, model_plural):
        """Generate the list page for a model."""
        objects = list(model.all_objects())
        
        title = model_plural.replace('_', ' ').title()
        content = f"# {title}\n\n"
        content += f"Total: {len(objects)} {model_plural}\n\n"
        
        # Generate table based on model type
        if model_snake == "task":
            content += self._generate_task_table_simple(objects)
            # content += self._generate_task_table(objects)  # Complex HTML version
        elif model_snake == "synset":
            content += self._generate_synset_table(objects)
        elif model_snake == "object":
            content += self._generate_object_table(objects)
        elif model_snake == "scene":
            content += self._generate_scene_table(objects)
        elif model_snake == "category":
            content += self._generate_category_table(objects)
        elif model_snake == "transition_rule":
            content += self._generate_transition_table(objects)
        elif model_snake == "attachment_pair":
            content += self._generate_attachment_table(objects)
        elif model_snake == "complaint_type":
            content += self._generate_complaint_type_table(objects)
        elif model_snake == "particle_system":
            content += self._generate_particle_table(objects)
        else:
            # Generic table
            content += "| Name | Details |\n"
            content += "|------|--------|\n"
            for obj in sorted(objects, key=lambda x: str(x)):
                pk = getattr(obj, model.Meta.pk)
                content += f"| [{obj}]({pk}.md) | View details |\n"
        
        output_path = self.output_dir / model_plural / "index.md"
        output_path.write_text(content)
        self.generated_files.append(str(output_path))
        
    def generate_detail_pages(self, model, model_snake, model_plural):
        """Generate detail pages for each object."""
        # Skip complaint types - these should not have detail pages
        if model_snake == "complaint_type":
            return
            
        objects = model.all_objects()
        pk_field = model.Meta.pk
        
        for obj in objects:
            pk_value = getattr(obj, pk_field)
            content = self._generate_detail_content(model_snake, obj)
            
            output_path = self.output_dir / model_plural / f"{pk_value}.md"
            output_path.write_text(content)
            self.generated_files.append(str(output_path))
            
    def _generate_detail_content(self, model_type, obj):
        """Generate detail page content based on model type."""
        if model_type == "task":
            return self._generate_task_detail(obj)
        elif model_type == "synset":
            return self._generate_synset_detail(obj)
        elif model_type == "object":
            return self._generate_object_detail(obj)
        elif model_type == "scene":
            return self._generate_scene_detail(obj)
        elif model_type == "category":
            return self._generate_category_detail(obj)
        elif model_type == "transition_rule":
            return self._generate_transition_detail(obj)
        elif model_type == "attachment_pair":
            return self._generate_attachment_detail(obj)
        elif model_type == "particle_system":
            return self._generate_particle_detail(obj)
        else:
            return f"# {obj}\n\nDetails for {obj}"
            
    def _generate_task_detail(self, task):
        """Generate task detail page."""
        content = f"# Task: {task.name}\n\n"
        
        # Status badges
        synset_color = self._status_to_badge_color(task.synset_state)
        scene_color = self._status_to_badge_color(task.scene_state)
        content += f"**Synset Status:** <span style='color: {synset_color}'>{task.synset_state.name}</span> | "
        content += f"**Scene Status:** <span style='color: {scene_color}'>{task.scene_state.name}</span>\n\n"
        
        # Required Synsets (with individual status)
        if task.synsets:
            content += "## Required Synsets\n\n"
            content += "| Synset | Status | Definition |\n"
            content += "|--------|--------|-----------|\n"
            for synset in sorted(task.synsets, key=lambda x: x.name):
                status_badge = self._status_to_badge(synset.state)
                definition = getattr(synset, 'definition', '') or ''
                # Truncate long definitions for table
                if len(definition) > 100:
                    definition = definition[:97] + "..."
                content += f"| [{synset.name}](../synsets/{synset.name}.md) | {status_badge} | {definition} |\n"
            content += "\n"
        
        # Scene Matching Status
        if hasattr(task, 'scene_matching_dict') and task.scene_matching_dict:
            content += "## Scene Matching Status\n\n"
            content += "| Scene | Status | Reason |\n"
            content += "|-------|--------|---------|\n"
            for scene, info in sorted(task.scene_matching_dict.items(), key=lambda x: x[0].name):
                # Handle both dict and object forms of info
                if isinstance(info, dict):
                    matched = info.get('matched', False)
                    reason = info.get('reason', '') if not matched else ""
                else:
                    matched = getattr(info, 'matched', False)
                    reason = getattr(info, 'reason', '') if not matched else ""
                
                status = "✅ Matched" if matched else "❌ Unmatched"
                content += f"| [{scene.name}](../scenes/{scene.name}.md) | {status} | {reason} |\n"
            content += "\n"
        
        # Transition Paths - Mermaid diagram
        if hasattr(task, 'transition_graph') and task.transition_graph:
            content += "## Transition Paths By Task Scope Objects\n\n"
            content += "```mermaid\n"
            content += "graph TD;\n"
            # Add nodes
            for node in task.transition_graph.nodes:
                # Slugify node names for Mermaid
                node_id = node.replace('.', '_').replace(' ', '_').replace('-', '_')
                content += f"    {node_id}({node});\n"
            # Add edges
            for edge in task.transition_graph.edges:
                source_id = edge[0].replace('.', '_').replace(' ', '_').replace('-', '_')
                target_id = edge[1].replace('.', '_').replace(' ', '_').replace('-', '_')
                content += f"    {source_id} --> {target_id};\n"
            content += "```\n\n"
        
        # Unreachable Goal Synsets
        if hasattr(task, 'goal_is_reachable') and not task.goal_is_reachable:
            if hasattr(task, 'unreachable_goal_synsets') and task.unreachable_goal_synsets:
                content += "## Unreachable Goal Synsets\n\n"
                for synset in task.unreachable_goal_synsets:
                    content += f"- [{synset.name}](../synsets/{synset.name}.md)\n"
                content += "\n"
        
        # Task definition (BDDL)
        if task.definition:
            content += "## Full Definition\n\n"
            content += f"```bddl\n{task.definition}\n```\n\n"
        
        # Debugging: Partial Transition Graph
        if hasattr(task, 'partial_transition_graph') and task.partial_transition_graph:
            content += "## Debugging: All Possible Recipes Resulting in Future Synsets\n\n"
            content += "```mermaid\n"
            content += "graph TD;\n"
            # Add nodes
            for node in task.partial_transition_graph.nodes:
                node_id = node.replace('.', '_').replace(' ', '_').replace('-', '_')
                content += f"    {node_id}({node});\n"
            # Add edges
            for edge in task.partial_transition_graph.edges:
                source_id = edge[0].replace('.', '_').replace(' ', '_').replace('-', '_')
                target_id = edge[1].replace('.', '_').replace(' ', '_').replace('-', '_')
                content += f"    {source_id} --> {target_id};\n"
            content += "```\n\n"
                
        return content
        
    def _generate_synset_detail(self, synset):
        """Generate comprehensive synset detail page."""
        content = f"# Synset: {synset.name}\n\n"
        
        # Status
        color = self._status_to_badge_color(synset.state)
        content += f"**Status:** <span style='color: {color}'>{synset.state.name}</span>\n\n"
        
        # Description/Definition
        if hasattr(synset, 'definition') and synset.definition:
            content += "## Description\n\n"
            content += f"{synset.definition}\n\n"
        
        # Parent Synsets Table
        if synset.parents:
            content += "## Parent Synsets\n\n"
            content += "| Name | Status | Definition |\n"
            content += "|------|--------|------------|\n"
            for parent in synset.parents:
                status_badge = self._status_to_badge(parent.state)
                definition = getattr(parent, 'definition', '') or '-'
                if len(definition) > 80:
                    definition = definition[:77] + "..."
                content += f"| [{parent.name}]({parent.name}.md) | {status_badge} | {definition} |\n"
            content += "\n"
            
        # Children Synsets Table
        if synset.children:
            content += "## Children Synsets\n\n"
            content += "| Name | Status | Definition | Direct Objects | Total Objects |\n"
            content += "|------|--------|------------|----------------|---------------|\n"
            for child in sorted(synset.children, key=lambda x: x.name):
                status_badge = self._status_to_badge(child.state)
                definition = getattr(child, 'definition', '') or '-'
                if len(definition) > 60:
                    definition = definition[:57] + "..."
                    
                direct_obj_count = 0
                if hasattr(child, 'direct_matching_objects'):
                    direct_obj_count = len(child.direct_matching_objects)
                    
                total_obj_count = 0
                if hasattr(child, 'matching_objects'):
                    total_obj_count = len(child.matching_objects)
                    
                content += f"| [{child.name}]({child.name}.md) | {status_badge} | {definition} | {direct_obj_count} | {total_obj_count} |\n"
            content += "\n"
            
        # Ancestor Synsets Table
        ancestors = []
        if hasattr(synset, 'ancestors'):
            ancestors = list(synset.ancestors)
        elif synset.parents:
            # Build ancestors recursively
            visited = set()
            def get_ancestors(node):
                if node in visited:
                    return []
                visited.add(node)
                result = []
                if hasattr(node, 'parents') and node.parents:
                    for parent in node.parents:
                        result.append(parent)
                        result.extend(get_ancestors(parent))
                return result
            ancestors = get_ancestors(synset)
            
        if ancestors:
            content += "## Ancestor Synsets\n\n"
            content += "| Name | Status | Definition |\n"
            content += "|------|--------|------------|\n"
            for ancestor in sorted(set(ancestors), key=lambda x: x.name):
                status_badge = self._status_to_badge(ancestor.state)
                definition = getattr(ancestor, 'definition', '') or '-'
                if len(definition) > 80:
                    definition = definition[:77] + "..."
                content += f"| [{ancestor.name}]({ancestor.name}.md) | {status_badge} | {definition} |\n"
            content += "\n"
            
        # Usage Table section
        content += "## Usage Types\n\n"
        content += "| Usage Type | Status |\n"
        content += "|------------|--------|\n"
        
        # Check for substance usage
        is_substance = getattr(synset, 'is_used_as_substance', False)
        content += f"| Used as Substance | {'✅ YES' if is_substance else '❌ NO'} |\n"
        
        # Check for non-substance usage
        is_non_substance = getattr(synset, 'is_used_as_non_substance', False)
        content += f"| Used as Non-Substance | {'✅ YES' if is_non_substance else '❌ NO'} |\n"
        
        # Check for fillable usage
        is_fillable = getattr(synset, 'is_used_as_fillable', False)
        content += f"| Used as Fillable | {'✅ YES' if is_fillable else '❌ NO'} |\n"
        
        # Add predicates that use this synset
        if hasattr(synset, 'used_in_predicates') and synset.used_in_predicates:
            for predicate in sorted(synset.used_in_predicates, key=lambda x: x.name):
                content += f"| Used in Predicate | {predicate.name} |\n"
        
        content += "\n"
        
        # Properties section
        has_pos_props = hasattr(synset, 'positive_properties') and synset.positive_properties
        has_neg_props = hasattr(synset, 'negative_properties') and synset.negative_properties
        has_property_names = hasattr(synset, 'property_names') and synset.property_names
        
        if has_pos_props or has_neg_props or has_property_names:
            content += "## Properties\n\n"
            
            if has_property_names:
                content += "**Property Names:** "
                content += ", ".join(sorted(synset.property_names))
                content += "\n\n"
                
            if has_pos_props:
                content += "**Positive Properties:**\n"
                for prop in sorted(synset.positive_properties):
                    content += f"- {prop}\n"
                content += "\n"
                
            if has_neg_props:
                content += "**Negative Properties:**\n"
                for prop in sorted(synset.negative_properties):
                    content += f"- {prop}\n"
                content += "\n"
                
                
        # Directly Mapped Categories Table
        if synset.categories:
            content += "## Directly Mapped Categories\n\n"
            content += "| Category | Objects | Synset Status |\n"
            content += "|----------|---------|---------------|\n"
            for cat in sorted(synset.categories, key=lambda x: x.name):
                obj_count = len(cat.objects) if hasattr(cat, 'objects') and cat.objects else 0
                synset_status = getattr(cat, 'synset_state', synset.state) if hasattr(cat, 'synset_state') else synset.state
                status_badge = self._status_to_badge(synset_status)
                content += f"| [{cat.name}](../categories/{cat.name}.md) | {obj_count} | {status_badge} |\n"
            content += "\n"
            
        # Directly Mapped Objects Table
        direct_objects = []
        if hasattr(synset, 'direct_matching_objects'):
            direct_objects = list(synset.direct_matching_objects)
        elif synset.categories:
            # Get objects from directly mapped categories
            for cat in synset.categories:
                if hasattr(cat, 'objects') and cat.objects:
                    direct_objects.extend(cat.objects)
                    
        if direct_objects:
            content += "## Directly Mapped Objects\n\n"
            content += "| Object | Category | Ready | Provider |\n"
            content += "|--------|----------|-------|----------|\n"
            for obj in sorted(set(direct_objects), key=lambda x: x.name)[:50]:  # Limit to 50
                category_name = obj.category.name if obj.category else '-'
                category_link = f"[{category_name}](../categories/{category_name}.md)" if obj.category else '-'
                ready_status = '✅' if getattr(obj, 'ready', False) else '⚠️'
                provider = getattr(obj, 'provider', '-')
                content += f"| [{obj.name}](../objects/{obj.name}.md) | {category_link} | {ready_status} | {provider} |\n"
            if len(direct_objects) > 50:
                content += f"\n*Showing first 50 of {len(direct_objects)} objects*\n"
            content += "\n"
            
        # All Descendant Objects Table  
        all_objects = []
        if hasattr(synset, 'matching_objects'):
            all_objects = list(synset.matching_objects)
        else:
            # Get all objects from this synset and all descendants
            visited_synsets = set()
            def collect_objects(syn):
                if syn in visited_synsets:
                    return []
                visited_synsets.add(syn)
                objects = []
                
                # Objects from categories of this synset
                if syn.categories:
                    for cat in syn.categories:
                        if hasattr(cat, 'objects') and cat.objects:
                            objects.extend(cat.objects)
                            
                # Objects from children synsets
                if hasattr(syn, 'children') and syn.children:
                    for child in syn.children:
                        objects.extend(collect_objects(child))
                return objects
            all_objects = collect_objects(synset)
            
        if all_objects:
            content += "## All Descendant Objects (Including Direct)\n\n"
            content += "| Object | Category | Ready | Provider | Bounding Box |\n"
            content += "|--------|----------|-------|----------|---------------|\n"
            for obj in sorted(set(all_objects), key=lambda x: x.name)[:100]:  # Limit to 100
                category_name = obj.category.name if obj.category else '-'
                category_link = f"[{category_name}](../categories/{category_name}.md)" if obj.category else '-'
                ready_status = '✅' if getattr(obj, 'ready', False) else '⚠️'
                provider = getattr(obj, 'provider', '-')
                
                bbox = '-'
                if hasattr(obj, 'bounding_box_size') and obj.bounding_box_size and len(obj.bounding_box_size) >= 3:
                    bbox = f"{obj.bounding_box_size[0]:.1f}×{obj.bounding_box_size[1]:.1f}×{obj.bounding_box_size[2]:.1f}"
                    
                content += f"| [{obj.name}](../objects/{obj.name}.md) | {category_link} | {ready_status} | {provider} | {bbox} |\n"
            if len(all_objects) > 100:
                content += f"\n*Showing first 100 of {len(all_objects)} objects*\n"
            content += "\n"
            
        # Tasks Table
        if hasattr(synset, 'tasks') and synset.tasks:
            content += "## Tasks\n\n"
            content += "| Task | Synset Status | Scene Status | Predicates | Features |\n"
            content += "|------|---------------|--------------|------------|----------|\n"
            for task in sorted(synset.tasks, key=lambda x: x.name):
                synset_badge = self._status_to_badge(task.synset_state)
                scene_badge = self._status_to_badge(task.scene_state)
                
                # Predicates
                predicates = []
                if hasattr(task, 'uses_predicates') and task.uses_predicates:
                    predicates = [pred.name for pred in task.uses_predicates[:3]]
                predicates_str = "<br/>".join(predicates) if predicates else '-'
                if hasattr(task, 'uses_predicates') and len(task.uses_predicates) > 3:
                    predicates_str += f"<br/>...+{len(task.uses_predicates)-3} more"
                
                # Features
                features = []
                if hasattr(task, 'uses_transition') and task.uses_transition:
                    features.append('transition')
                if hasattr(task, 'uses_visual_substance') and task.uses_visual_substance:
                    features.append('visual_substance')
                if hasattr(task, 'uses_physical_substance') and task.uses_physical_substance:
                    features.append('physical_substance')
                if hasattr(task, 'uses_attachment') and task.uses_attachment:
                    features.append('attachment')
                if hasattr(task, 'uses_cloth') and task.uses_cloth:
                    features.append('cloth')
                features_str = "<br/>".join(features) if features else '-'
                
                content += f"| [{task.name}](../tasks/{task.name}.md) | {synset_badge} | {scene_badge} | {predicates_str} | {features_str} |\n"
            content += "\n"
            
        # Transitions Table
        transitions = []
        if hasattr(synset, 'transitions'):
            transitions = list(synset.transitions)
        elif hasattr(synset, 'transition_rules'):
            transitions = list(synset.transition_rules)
            
        if transitions:
            content += "## Transitions\n\n"
            content += "| From State | To State | Condition | Rule |\n"
            content += "|------------|----------|-----------|------|\n"
            for transition in transitions:
                from_state = getattr(transition, 'from_state', getattr(transition, 'initial_state', '-'))
                to_state = getattr(transition, 'to_state', getattr(transition, 'final_state', '-'))
                condition = getattr(transition, 'condition', getattr(transition, 'predicate', '-'))
                rule = getattr(transition, 'rule', getattr(transition, 'name', '-'))
                content += f"| {from_state} | {to_state} | {condition} | {rule} |\n"
            content += "\n"
            
        # Related Synsets
        related_synsets = set()
        if hasattr(synset, 'related_synsets'):
            related_synsets.update(synset.related_synsets)
        # Add synsets that share categories with this one
        if synset.categories:
            for cat in synset.categories:
                if hasattr(cat, 'synset') and cat.synset and cat.synset != synset:
                    related_synsets.add(cat.synset)
                    
        if related_synsets:
            content += "## Related Synsets\n\n"
            content += "| Name | Status | Relationship | Common Categories |\n"
            content += "|------|--------|--------------|-------------------|\n"
            for related in sorted(related_synsets, key=lambda x: x.name):
                status_badge = self._status_to_badge(related.state)
                
                # Determine relationship
                relationship = "Related"
                if related in (synset.parents or []):
                    relationship = "Parent"
                elif related in (synset.children or []):
                    relationship = "Child"
                elif hasattr(synset, 'siblings') and related in (synset.siblings or []):
                    relationship = "Sibling"
                    
                # Find common categories
                common_cats = []
                if synset.categories and related.categories:
                    common_cats = [cat.name for cat in synset.categories if cat in related.categories]
                common_cats_str = ", ".join(common_cats[:3])
                if len(common_cats) > 3:
                    common_cats_str += f", ...+{len(common_cats)-3}"
                if not common_cats_str:
                    common_cats_str = "-"
                    
                content += f"| [{related.name}]({related.name}.md) | {status_badge} | {relationship} | {common_cats_str} |\n"
            content += "\n"
            
        # Hierarchy diagram
        if hasattr(synset, 'subgraph') and synset.subgraph:
            content += "## Hierarchy Diagram\n\n"
            content += self._generate_mermaid_graph(synset.subgraph, synset)
            content += "\n"
        elif synset.parents or synset.children:
            # Generate a simple hierarchy diagram if no subgraph
            content += "## Hierarchy Diagram\n\n"
            content += "```mermaid\ngraph TD\n"
            
            # Add parents
            for parent in (synset.parents or []):
                parent_id = self._sanitize_mermaid_id(parent.name)
                synset_id = self._sanitize_mermaid_id(synset.name)
                content += f"    {parent_id}[{parent.name}] --> {synset_id}[{synset.name}]\n"
                
            # Add children  
            for child in (synset.children or []):
                child_id = self._sanitize_mermaid_id(child.name)
                synset_id = self._sanitize_mermaid_id(synset.name)
                content += f"    {synset_id}[{synset.name}] --> {child_id}[{child.name}]\n"
                
            content += "```\n\n"
            
        return content
        
    def _generate_object_detail(self, obj):
        """Generate object detail page."""
        content = f"# Object: {obj.name}\n\n"
        
        # Status and metadata
        if obj.ready:
            content += "**Status:** ✅ Ready\n\n"
        else:
            content += "**Status:** ⚠️ Has Unresolved Complaints\n\n"
            
        if obj.category:
            content += f"**Category:** [{obj.category.name}](../categories/{obj.category.name}.md)\n\n"
        elif obj.particle_system:
            content += f"**Particle System:** [{obj.particle_system.name}](../particle_systems/{obj.particle_system.name}.md)\n\n"
            
        content += f"**Original Category:** {obj.original_category_name} | **Provider:** {obj.provider}\n\n"
        
        # Bounding box
        if obj.bounding_box_size:
            content += f"**Bounding Box Size (meters):** {obj.bounding_box_size[0]:.2f} × {obj.bounding_box_size[1]:.2f} × {obj.bounding_box_size[2]:.2f}\n\n"
            
        # Video/Image
        if obj.image_url:
            content += "## Preview\n\n"
            content += f'<video autoplay muted loop playsinline src="{obj.image_url}" style="max-width: 100%; height: auto;"></video>\n\n'
            
        # Meta links
        if obj.meta_links:
            content += "## Meta Links\n\n"
            content += "| Name |\n"
            content += "|------|\n"
            for ml in obj.meta_links:
                content += f"| {ml.name} |\n"
            content += "\n"
            
        # Scene/Room Usage - fix data parsing
        if obj.roomobjects:
            content += "## Scene/Room Usage\n\n"
            content += "| Scene | Room |\n"
            content += "|-------|------|\n"
            for ro in obj.roomobjects:
                scene_name = ro.room.scene.name if ro.room and ro.room.scene else '-'
                room_name = ro.room.name if ro.room else '-'
                
                # Only create links if we have valid scene names
                if scene_name != '-':
                    scene_link = f"[{scene_name}](../scenes/{scene_name}.md)"
                else:
                    scene_link = scene_name
                    
                content += f"| {scene_link} | {room_name} |\n"
            content += "\n"
            
        # Attachment Pairs
        attachment_pairs = []
        if hasattr(obj, 'attachment_pairs') and obj.attachment_pairs:
            attachment_pairs = obj.attachment_pairs
        # Also check if this object is referenced in any attachment pairs
        if hasattr(obj, '_attachment_pairs_as_parent') and obj._attachment_pairs_as_parent:
            attachment_pairs.extend(obj._attachment_pairs_as_parent)
        if hasattr(obj, '_attachment_pairs_as_child') and obj._attachment_pairs_as_child:
            attachment_pairs.extend(obj._attachment_pairs_as_child)
            
        if attachment_pairs:
            content += "## Attachment Pairs\n\n"
            content += "| Parent Object | Child Object | Attachment Type | Condition |\n"
            content += "|---------------|---------------|-----------------|----------|\n"
            for ap in sorted(set(attachment_pairs), key=lambda x: getattr(x, 'name', str(x))):
                parent_name = '-'
                child_name = '-'
                attachment_type = '-'
                condition = '-'
                
                if hasattr(ap, 'parent') and ap.parent:
                    parent_name = ap.parent.name if hasattr(ap.parent, 'name') else str(ap.parent)
                    parent_name = f"[{parent_name}](../objects/{parent_name}.md)" if parent_name != '-' else '-'
                elif hasattr(ap, 'parent_name'):
                    parent_name = f"[{ap.parent_name}](../objects/{ap.parent_name}.md)"
                    
                if hasattr(ap, 'child') and ap.child:
                    child_name = ap.child.name if hasattr(ap.child, 'name') else str(ap.child)
                    child_name = f"[{child_name}](../objects/{child_name}.md)" if child_name != '-' else '-'
                elif hasattr(ap, 'child_name'):
                    child_name = f"[{ap.child_name}](../objects/{ap.child_name}.md)"
                    
                if hasattr(ap, 'attachment_type'):
                    attachment_type = str(ap.attachment_type)
                elif hasattr(ap, 'type'):
                    attachment_type = str(ap.type)
                    
                if hasattr(ap, 'condition'):
                    condition = str(ap.condition)
                elif hasattr(ap, 'predicate'):
                    condition = str(ap.predicate)
                    
                content += f"| {parent_name} | {child_name} | {attachment_type} | {condition} |\n"
            content += "\n"
            
        # Synset hierarchy
        if obj.owner and obj.owner.synset:
            content += "## Synset Hierarchy\n\n"
            content += f"This object belongs to synset: [{obj.owner.synset.name}](../synsets/{obj.owner.synset.name}.md)\n\n"
            if obj.owner.synset.subgraph:
                content += self._generate_mermaid_graph(obj.owner.synset.subgraph, obj.owner.synset)
                content += "\n"
                
        return content
        
    def _generate_category_detail(self, category):
        """Generate comprehensive category detail page."""
        content = f"# Category: {category.name}\n\n"
        
        # Basic Information
        if hasattr(category, 'synset') and category.synset:
            synset_badge = self._status_to_badge(category.synset.state)
            content += f"**Synset:** [{category.synset.name}](../synsets/{category.synset.name}.md) {synset_badge}\n\n"
        
        if hasattr(category, 'description') and category.description:
            content += f"**Description:** {category.description}\n\n"
            
        # Object Statistics
        object_count = len(category.objects) if hasattr(category, 'objects') and category.objects else 0
        ready_count = 0
        if category.objects:
            ready_count = sum(1 for obj in category.objects if getattr(obj, 'ready', False))
        
        content += f"**Objects:** {object_count} total, {ready_count} ready\n\n"
        
        # Object Images Gallery
        if category.objects:
            content += "## Object Images\n\n"
            content += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">\n'
            
            image_count = 0
            for obj in sorted(category.objects, key=lambda x: x.name):
                if hasattr(obj, 'image_url') and obj.image_url and image_count < 20:  # Limit to 20 images
                    ready_icon = "✅" if getattr(obj, 'ready', False) else "⚠️"
                    content += f'''<div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; text-align: center;">
    <video autoplay muted loop playsinline src="{obj.image_url}" style="width: 100%; height: 150px; object-fit: cover; border-radius: 4px;"></video>
    <div style="margin-top: 8px; font-size: 0.9em;">
        <a href="../objects/{obj.name}.md" style="font-weight: bold; text-decoration: none;">{obj.name}</a><br/>
        <span style="color: #666;">{ready_icon} {getattr(obj, 'provider', 'Unknown')}</span>
    </div>
</div>
'''
                    image_count += 1
                    
            content += '</div>\n\n'
            if image_count == 20 and len(category.objects) > 20:
                content += f"*Showing first 20 of {len(category.objects)} objects with images*\n\n"
        
        # Comprehensive Objects Table
        if category.objects:
            content += "## Objects\n\n"
            content += "| Object | Category/System | Synset | Bounding Box | Meta Links | Used Rooms | Ready |\n"
            content += "|--------|-----------------|--------|--------------|------------|------------|-------|\n"
            
            for obj in sorted(category.objects, key=lambda x: x.name):
                # Object name
                obj_name = f"[{obj.name}](../objects/{obj.name}.md)"
                
                # Category or Particle System
                if hasattr(obj, 'owner') and obj.owner:
                    owner_name = obj.owner.name
                    category_cell = f'[{owner_name}](../categories/{owner_name}.md)'
                elif obj.category:
                    category_cell = f'[{obj.category.name}](../categories/{obj.category.name}.md)'
                elif hasattr(obj, 'particle_system') and obj.particle_system:
                    category_cell = f'[{obj.particle_system.name}](../particle_systems/{obj.particle_system.name}.md)'
                else:
                    category_cell = "-"
                
                # Synset
                synset_cell = "-"
                if hasattr(obj, 'owner') and obj.owner and hasattr(obj.owner, 'synset') and obj.owner.synset:
                    synset_cell = f'[{obj.owner.synset.name}](../synsets/{obj.owner.synset.name}.md)'
                elif obj.category and hasattr(obj.category, 'synset') and obj.category.synset:
                    synset_cell = f'[{obj.category.synset.name}](../synsets/{obj.category.synset.name}.md)'
                elif hasattr(obj, 'particle_system') and obj.particle_system and hasattr(obj.particle_system, 'synset') and obj.particle_system.synset:
                    synset_cell = f'[{obj.particle_system.synset.name}](../synsets/{obj.particle_system.synset.name}.md)'
                
                # Bounding Box
                bbox_cell = "-"
                if hasattr(obj, 'bounding_box_size') and obj.bounding_box_size and len(obj.bounding_box_size) >= 3:
                    bbox_cell = f"{obj.bounding_box_size[0]:.1f}×{obj.bounding_box_size[1]:.1f}×{obj.bounding_box_size[2]:.1f}"
                
                # Meta Links
                meta_links_count = 0
                if hasattr(obj, 'meta_links') and obj.meta_links:
                    meta_links_count = len(obj.meta_links)
                meta_links_cell = str(meta_links_count) if meta_links_count > 0 else "-"
                
                # Used Rooms
                rooms_count = 0
                if hasattr(obj, 'roomobjects') and obj.roomobjects:
                    rooms_count = len(obj.roomobjects)
                
                # Ready Status
                ready_cell = '✅' if getattr(obj, 'ready', False) else '⚠️'
                
                content += f"| {obj_name} | {category_cell} | {synset_cell} | {bbox_cell} | {meta_links_cell} | {rooms_count} | {ready_cell} |\n"
            content += "\n"
        
        # Ancestor Synsets Table
        ancestors = []
        if category.synset:
            if hasattr(category.synset, 'ancestors'):
                ancestors = list(category.synset.ancestors)
            elif hasattr(category.synset, 'parents'):
                # Build ancestors recursively
                visited = set()
                def get_ancestors(syn):
                    if syn in visited:
                        return []
                    visited.add(syn)
                    result = []
                    if hasattr(syn, 'parents') and syn.parents:
                        for parent in syn.parents:
                            result.append(parent)
                            result.extend(get_ancestors(parent))
                    return result
                ancestors = get_ancestors(category.synset)
        
        if ancestors:
            content += "## Ancestor Synsets\n\n"
            content += "| Synset | Status | Definition | Direct Objects | Total Objects |\n"
            content += "|--------|--------|------------|----------------|---------------|\n"
            for ancestor in sorted(set(ancestors), key=lambda x: x.name):
                status_badge = self._status_to_badge(ancestor.state)
                definition = getattr(ancestor, 'definition', '') or '-'
                if len(definition) > 80:
                    definition = definition[:77] + "..."
                    
                direct_obj_count = 0
                if hasattr(ancestor, 'direct_matching_objects'):
                    direct_obj_count = len(ancestor.direct_matching_objects)
                elif hasattr(ancestor, 'categories'):
                    for cat in ancestor.categories:
                        if hasattr(cat, 'objects') and cat.objects:
                            direct_obj_count += len(cat.objects)
                            
                total_obj_count = 0
                if hasattr(ancestor, 'matching_objects'):
                    total_obj_count = len(ancestor.matching_objects)
                    
                content += f"| [{ancestor.name}](../synsets/{ancestor.name}.md) | {status_badge} | {definition} | {direct_obj_count} | {total_obj_count} |\n"
            content += "\n"
        
        # Related Synsets Graph
        if category.synset and (hasattr(category.synset, 'subgraph') and category.synset.subgraph):
            content += "## Related Synsets Graph\n\n"
            content += self._generate_mermaid_graph(category.synset.subgraph, category.synset)
            content += "\n"
        elif category.synset and (hasattr(category.synset, 'parents') or hasattr(category.synset, 'children')):
            # Generate a simple hierarchy diagram
            content += "## Related Synsets Graph\n\n"
            content += "```mermaid\ngraph TD\n"
            
            # Add parents
            if hasattr(category.synset, 'parents') and category.synset.parents:
                for parent in category.synset.parents:
                    parent_id = self._sanitize_mermaid_id(parent.name)
                    synset_id = self._sanitize_mermaid_id(category.synset.name)
                    content += f"    {parent_id}[{parent.name}] --> {synset_id}[{category.synset.name}]\n"
                    
            # Add children  
            if hasattr(category.synset, 'children') and category.synset.children:
                for child in category.synset.children:
                    child_id = self._sanitize_mermaid_id(child.name)
                    synset_id = self._sanitize_mermaid_id(category.synset.name)
                    content += f"    {synset_id}[{category.synset.name}] --> {child_id}[{child.name}]\n"
                    
            # Highlight current synset
            synset_id = self._sanitize_mermaid_id(category.synset.name)
            content += f"    classDef current fill:#e1f5fe;\n"
            content += f"    class {synset_id} current;\n"
            content += "```\n\n"
        
        return content
        
    def _generate_scene_detail(self, scene):
        """Generate scene detail page."""
        content = f"# Scene: {scene.name}\n\n"
        
        # Rooms Table - following the exact structure from HTML template
        if hasattr(scene, 'rooms') and scene.rooms:
            content += "## Rooms\n\n"
            content += "| Room Name | Objects | Object Counts |\n"
            content += "|-----------|---------|---------------|\n"
            
            for room in sorted(scene.rooms, key=lambda x: x.name):
                room_name = room.name
                
                # Process room objects - each roomobject has .object and .count
                objects_list = []
                counts_list = []
                
                if hasattr(room, 'roomobjects') and room.roomobjects:
                    for roomobject in room.roomobjects:
                        # Get object name and create link
                        if hasattr(roomobject, 'object') and roomobject.object:
                            obj_name = roomobject.object.name if hasattr(roomobject.object, 'name') else str(roomobject.object)
                            objects_list.append(f"[{obj_name}](../objects/{obj_name}.md)")
                            
                            # Get count for this object
                            count = roomobject.count if hasattr(roomobject, 'count') else 1
                            counts_list.append(str(count))
                
                # Join with line breaks for multi-line cell content
                objects_str = "<br/>".join(objects_list) if objects_list else "-"
                counts_str = "<br/>".join(counts_list) if counts_list else "-"
                
                content += f"| {room_name} | {objects_str} | {counts_str} |\n"
            content += "\n"
            
        return content
        
    def _generate_transition_detail(self, rule):
        """Generate comprehensive transition rule detail page."""
        content = f"# Transition Rule: {rule.name}\n\n"
        
        # Rule type and description
        if hasattr(rule, 'rule_type'):
            content += f"**Type:** {rule.rule_type}\n\n"
        if hasattr(rule, 'description') and rule.description:
            content += f"**Description:** {rule.description}\n\n"
            
        # Input Synsets Table
        if hasattr(rule, 'input_synsets') and rule.input_synsets:
            content += "## Input Synsets\n\n"
            content += "| Synset | Status | Definition | Categories | Objects |\n"
            content += "|--------|--------|------------|------------|----------|\n"
            for synset in sorted(rule.input_synsets, key=lambda x: x.name):
                status_badge = self._status_to_badge(synset.state)
                definition = getattr(synset, 'definition', '') or '-'
                if len(definition) > 80:
                    definition = definition[:77] + "..."
                    
                # Categories count
                categories_count = len(synset.categories) if hasattr(synset, 'categories') and synset.categories else 0
                
                # Objects count
                objects_count = 0
                if hasattr(synset, 'matching_objects') and synset.matching_objects:
                    objects_count = len(synset.matching_objects)
                elif hasattr(synset, 'categories') and synset.categories:
                    for cat in synset.categories:
                        if hasattr(cat, 'objects') and cat.objects:
                            objects_count += len(cat.objects)
                
                content += f"| [{synset.name}](../synsets/{synset.name}.md) | {status_badge} | {definition} | {categories_count} | {objects_count} |\n"
            content += "\n"
            
        # Machine Synsets Table (the missing table)
        if hasattr(rule, 'machine_synsets') and rule.machine_synsets:
            content += "## Machine Synsets\n\n"
            content += "| Synset | Status | Definition | Role | Categories |\n"
            content += "|--------|--------|------------|------|------------|\n"
            for synset in sorted(rule.machine_synsets, key=lambda x: x.name):
                status_badge = self._status_to_badge(synset.state)
                definition = getattr(synset, 'definition', '') or '-'
                if len(definition) > 80:
                    definition = definition[:77] + "..."
                    
                # Role in transition (if available)
                role = getattr(synset, 'transition_role', 'Machine') if hasattr(synset, 'transition_role') else 'Machine'
                
                # Categories count
                categories_count = len(synset.categories) if hasattr(synset, 'categories') and synset.categories else 0
                
                content += f"| [{synset.name}](../synsets/{synset.name}.md) | {status_badge} | {definition} | {role} | {categories_count} |\n"
            content += "\n"
            
        # Output Synsets Table
        if hasattr(rule, 'output_synsets') and rule.output_synsets:
            content += "## Output Synsets\n\n"
            content += "| Synset | Status | Definition | Categories | Objects |\n"
            content += "|--------|--------|------------|------------|----------|\n"
            for synset in sorted(rule.output_synsets, key=lambda x: x.name):
                status_badge = self._status_to_badge(synset.state)
                definition = getattr(synset, 'definition', '') or '-'
                if len(definition) > 80:
                    definition = definition[:77] + "..."
                    
                # Categories count
                categories_count = len(synset.categories) if hasattr(synset, 'categories') and synset.categories else 0
                
                # Objects count
                objects_count = 0
                if hasattr(synset, 'matching_objects') and synset.matching_objects:
                    objects_count = len(synset.matching_objects)
                elif hasattr(synset, 'categories') and synset.categories:
                    for cat in synset.categories:
                        if hasattr(cat, 'objects') and cat.objects:
                            objects_count += len(cat.objects)
                
                content += f"| [{synset.name}](../synsets/{synset.name}.md) | {status_badge} | {definition} | {categories_count} | {objects_count} |\n"
            content += "\n"
            
        # Transition Rule Graph (the missing graph)
        content += "## Transition Rule Graph\n\n"
        content += "```mermaid\ngraph LR\n"
        
        # Add input synsets
        if hasattr(rule, 'input_synsets') and rule.input_synsets:
            for synset in rule.input_synsets:
                synset_id = self._sanitize_mermaid_id(synset.name)
                content += f"    {synset_id}[{synset.name}]\n"
                content += f"    {synset_id} --> RULE\n"
        
        # Add rule node
        content += f"    RULE[{rule.name}]\n"
        
        # Add machine synsets (if any)
        if hasattr(rule, 'machine_synsets') and rule.machine_synsets:
            for synset in rule.machine_synsets:
                synset_id = self._sanitize_mermaid_id(synset.name)
                content += f"    {synset_id}[{synset.name}]\n"
                content += f"    {synset_id} -.-> RULE\n"  # Dotted line for machines
        
        # Add output synsets
        if hasattr(rule, 'output_synsets') and rule.output_synsets:
            for synset in rule.output_synsets:
                synset_id = self._sanitize_mermaid_id(synset.name)
                content += f"    {synset_id}[{synset.name}]\n"
                content += f"    RULE --> {synset_id}\n"
        
        # Style the graph
        content += "    classDef inputSynset fill:#e3f2fd;\n"
        content += "    classDef machineSynset fill:#fff3e0;\n"
        content += "    classDef outputSynset fill:#e8f5e8;\n"
        content += "    classDef ruleNode fill:#fce4ec;\n"
        
        if hasattr(rule, 'input_synsets') and rule.input_synsets:
            input_ids = [self._sanitize_mermaid_id(s.name) for s in rule.input_synsets]
            content += f"    class {','.join(input_ids)} inputSynset;\n"
            
        if hasattr(rule, 'machine_synsets') and rule.machine_synsets:
            machine_ids = [self._sanitize_mermaid_id(s.name) for s in rule.machine_synsets]
            content += f"    class {','.join(machine_ids)} machineSynset;\n"
            
        if hasattr(rule, 'output_synsets') and rule.output_synsets:
            output_ids = [self._sanitize_mermaid_id(s.name) for s in rule.output_synsets]
            content += f"    class {','.join(output_ids)} outputSynset;\n"
            
        content += f"    class RULE ruleNode;\n"
        content += "```\n\n"
        
        # Conditions
        if hasattr(rule, 'conditions') and rule.conditions:
            content += "## Conditions\n\n"
            content += f"```\n{rule.conditions}\n```\n\n"
        
        # Related Rules (if available)
        if hasattr(rule, 'related_rules') and rule.related_rules:
            content += "## Related Rules\n\n"
            content += "| Rule | Shared Synsets | Rule Type |\n"
            content += "|------|----------------|----------|\n"
            for related_rule in sorted(rule.related_rules, key=lambda x: x.name):
                shared_count = 0
                # Calculate shared synsets (simplified)
                rule_type = getattr(related_rule, 'rule_type', 'Unknown')
                content += f"| [{related_rule.name}]({related_rule.name}.md) | {shared_count} | {rule_type} |\n"
            content += "\n"
            
        return content
        
    def _generate_attachment_detail(self, pair):
        """Generate comprehensive attachment pair detail page."""
        content = f"# Attachment Pair: {pair.name}\n\n"
        
        # Basic information
        if hasattr(pair, 'description') and pair.description:
            content += f"**Description:** {pair.description}\n\n"
            
        # Attachment type and relationship
        if hasattr(pair, 'attachment_type'):
            content += f"**Attachment Type:** {pair.attachment_type}\n\n"
        if hasattr(pair, 'relationship_type'):
            content += f"**Relationship Type:** {pair.relationship_type}\n\n"
        
        # Synset information
        if hasattr(pair, 'male_synset') and pair.male_synset:
            content += f"**Male Synset:** [{pair.male_synset.name}](../synsets/{pair.male_synset.name}.md)\n\n"
            
        if hasattr(pair, 'female_synset') and pair.female_synset:
            content += f"**Female Synset:** [{pair.female_synset.name}](../synsets/{pair.female_synset.name}.md)\n\n"
        
        # Female Objects Table
        female_objects = []
        if hasattr(pair, 'female_objects') and pair.female_objects:
            female_objects = pair.female_objects
        elif hasattr(pair, 'female_synset') and pair.female_synset and hasattr(pair.female_synset, 'matching_objects'):
            female_objects = list(pair.female_synset.matching_objects)
            
        if female_objects:
            content += "## Female Objects\n\n"
            content += "| Object | Category | Synset | Ready | Provider |\n"
            content += "|--------|----------|--------|-------|----------|\n"
            
            for obj in sorted(female_objects, key=lambda x: getattr(x, 'name', str(x))):
                obj_name = getattr(obj, 'name', str(obj))
                
                # Category
                category_name = "-"
                if hasattr(obj, 'category') and obj.category:
                    category_name = f"[{obj.category.name}](../categories/{obj.category.name}.md)"
                elif hasattr(obj, 'owner') and obj.owner:
                    category_name = f"[{obj.owner.name}](../categories/{obj.owner.name}.md)"
                
                # Synset
                synset_name = "-"
                if hasattr(obj, 'category') and obj.category and hasattr(obj.category, 'synset') and obj.category.synset:
                    synset_name = f"[{obj.category.synset.name}](../synsets/{obj.category.synset.name}.md)"
                elif hasattr(obj, 'owner') and obj.owner and hasattr(obj.owner, 'synset') and obj.owner.synset:
                    synset_name = f"[{obj.owner.synset.name}](../synsets/{obj.owner.synset.name}.md)"
                
                # Ready status
                ready_status = '✅' if getattr(obj, 'ready', False) else '⚠️'
                
                # Provider
                provider = getattr(obj, 'provider', '-')
                
                content += f"| [{obj_name}](../objects/{obj_name}.md) | {category_name} | {synset_name} | {ready_status} | {provider} |\n"
            content += "\n"
        
        # Male Objects Table
        male_objects = []
        if hasattr(pair, 'male_objects') and pair.male_objects:
            male_objects = pair.male_objects
        elif hasattr(pair, 'male_synset') and pair.male_synset and hasattr(pair.male_synset, 'matching_objects'):
            male_objects = list(pair.male_synset.matching_objects)
            
        if male_objects:
            content += "## Male Objects\n\n"
            content += "| Object | Category | Synset | Ready | Provider |\n"
            content += "|--------|----------|--------|-------|----------|\n"
            
            for obj in sorted(male_objects, key=lambda x: getattr(x, 'name', str(x))):
                obj_name = getattr(obj, 'name', str(obj))
                
                # Category
                category_name = "-"
                if hasattr(obj, 'category') and obj.category:
                    category_name = f"[{obj.category.name}](../categories/{obj.category.name}.md)"
                elif hasattr(obj, 'owner') and obj.owner:
                    category_name = f"[{obj.owner.name}](../categories/{obj.owner.name}.md)"
                
                # Synset
                synset_name = "-"
                if hasattr(obj, 'category') and obj.category and hasattr(obj.category, 'synset') and obj.category.synset:
                    synset_name = f"[{obj.category.synset.name}](../synsets/{obj.category.synset.name}.md)"
                elif hasattr(obj, 'owner') and obj.owner and hasattr(obj.owner, 'synset') and obj.owner.synset:
                    synset_name = f"[{obj.owner.synset.name}](../synsets/{obj.owner.synset.name}.md)"
                
                # Ready status
                ready_status = '✅' if getattr(obj, 'ready', False) else '⚠️'
                
                # Provider
                provider = getattr(obj, 'provider', '-')
                
                content += f"| [{obj_name}](../objects/{obj_name}.md) | {category_name} | {synset_name} | {ready_status} | {provider} |\n"
            content += "\n"
            
        return content
        
    def _generate_particle_detail(self, particle_system):
        """Generate comprehensive particle system detail page."""
        content = f"# {particle_system.name}\n\n"
        
        # Summary section matching the old template structure
        content += f"## Particle System Details\n\n"
        
        # Synset Information (prominent display like in old template)
        if hasattr(particle_system, 'synset') and particle_system.synset:
            synset_badge = self._status_to_badge(particle_system.synset.state)
            content += f"**Synset:** [{particle_system.synset.name}](../synsets/{particle_system.synset.name}.md) {synset_badge}\n\n"
        else:
            content += f"**Synset:** None\n\n"
        
        # Basic properties
        if hasattr(particle_system, 'description') and particle_system.description:
            content += f"**Description:** {particle_system.description}\n\n"
            
        # Type and system properties
        if hasattr(particle_system, 'system_type'):
            content += f"**System Type:** {particle_system.system_type}\n\n"
        if hasattr(particle_system, 'physics_type'):
            content += f"**Physics Type:** {particle_system.physics_type}\n\n"
        
        # Additional properties from the particle system
        if hasattr(particle_system, 'particle_count'):
            content += f"**Particle Count:** {particle_system.particle_count}\n\n"
        if hasattr(particle_system, 'mass'):
            content += f"**Mass:** {particle_system.mass}\n\n"
            
        # Particle Model Images Gallery
        particle_models = []
        if hasattr(particle_system, 'particle_models') and particle_system.particle_models:
            particle_models = particle_system.particle_models
        elif hasattr(particle_system, 'models') and particle_system.models:
            particle_models = particle_system.models
            
        if particle_models:
            content += "## Particle Model Images\n\n"
            content += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; margin: 20px 0;">\n'
            
            image_count = 0
            for model in particle_models[:20]:  # Limit to 20 images
                if hasattr(model, 'image_url') and model.image_url:
                    model_name = getattr(model, 'name', f'Model {image_count + 1}')
                    # Check if this is a video file (based on file extension)
                    is_video = model.image_url.lower().endswith(('.mp4', '.webm', '.ogg', '.mov'))
                    
                    if is_video:
                        content += f'''<div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; text-align: center;">
    <div style="margin-bottom: 8px; font-weight: bold;">
        <a href="../objects/{model_name}.html" style="color: #007bff; text-decoration: none;">{model_name}</a>
    </div>
    <video autoplay muted loop playsinline src="{model.image_url}" style="width: 100%; height: 150px; object-fit: cover; border-radius: 4px;" alt="{model_name} video"></video>
</div>
'''
                    else:
                        content += f'''<div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; text-align: center;">
    <div style="margin-bottom: 8px; font-weight: bold;">
        <a href="../objects/{model_name}.html" style="color: #007bff; text-decoration: none;">{model_name}</a>
    </div>
    <img src="{model.image_url}" style="width: 100%; height: 150px; object-fit: cover; border-radius: 4px;" alt="{model_name}"/>
</div>
'''
                    image_count += 1
                elif particle_models:  # Show entry even without image
                    model_name = getattr(model, 'name', f'Model {image_count + 1}')
                    content += f'''<div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; text-align: center; background-color: #f8f9fa;">
    <div style="margin-bottom: 8px; font-weight: bold;">
        <a href="../objects/{model_name}.html" style="color: #007bff; text-decoration: none;">{model_name}</a>
    </div>
    <div style="height: 150px; display: flex; align-items: center; justify-content: center; color: #6c757d;">
        No image available
    </div>
</div>
'''
                    image_count += 1
                    
            content += '</div>\n\n'
            if len(particle_models) > 20:
                content += f"*Showing first 20 of {len(particle_models)} particle models*\n\n"
            elif len(particle_models) == 0:
                content += '<div style="text-align: center; padding: 20px; color: #6c757d;">No particle models found</div>\n\n'
        
        # Particle Models Table
        if particle_models:
            content += "## Particle Models\n\n"
            content += "| Model Name | Category | Ready | Provider | Type | Properties |\n"
            content += "|------------|----------|-------|----------|------|------------|\n"
            
            for model in sorted(particle_models, key=lambda x: getattr(x, 'name', str(x))):
                model_name = getattr(model, 'name', 'Unknown')
                
                # Category information
                category_name = "-"
                if hasattr(model, 'category') and model.category:
                    category_name = f"[{model.category.name}](../categories/{model.category.name}.md)"
                elif hasattr(model, 'owner') and model.owner:
                    category_name = f"[{model.owner.name}](../categories/{model.owner.name}.md)"
                
                # Ready status
                ready_status = '✅' if getattr(model, 'ready', False) else '⚠️'
                
                # Provider
                provider = getattr(model, 'provider', '-')
                
                # Type and properties  
                model_type = getattr(model, 'model_type', getattr(model, 'type', '-'))
                properties = getattr(model, 'properties', '-')
                if isinstance(properties, (list, tuple)):
                    properties = ", ".join(map(str, properties))
                elif isinstance(properties, dict):
                    prop_list = []
                    for k, v in properties.items():
                        prop_list.append(f"{k}: {v}")
                    properties = ", ".join(prop_list)
                
                # Add links to model detail pages
                if hasattr(model, 'name'):
                    model_link = f"[{model_name}](../objects/{model_name}.md)"
                else:
                    model_link = model_name
                
                content += f"| {model_link} | {category_name} | {ready_status} | {provider} | {model_type} | {properties} |\n"
            content += "\n"
        else:
            content += "## Particle Models\n\n"
            content += '<div style="text-align: center; padding: 20px; color: #6c757d;">No particle models found</div>\n\n'
        
        # Parent Synsets Table
        if hasattr(particle_system, 'synset') and particle_system.synset and hasattr(particle_system.synset, 'parents') and particle_system.synset.parents:
            content += "## Parent Synsets\n\n"
            content += "| Name | Status | Definition |\n"
            content += "|------|--------|------------|\n"
            for parent in particle_system.synset.parents:
                status_badge = self._status_to_badge(parent.state)
                definition = getattr(parent, 'definition', '') or '-'
                if len(definition) > 80:
                    definition = definition[:77] + "..."
                content += f"| [{parent.name}](../synsets/{parent.name}.md) | {status_badge} | {definition} |\n"
            content += "\n"
            
        # Children Synsets Table
        if hasattr(particle_system, 'synset') and particle_system.synset and hasattr(particle_system.synset, 'children') and particle_system.synset.children:
            content += "## Children Synsets\n\n"
            content += "| Name | Status | Definition | Categories | Objects |\n"
            content += "|------|--------|------------|------------|----------|\n"
            for child in sorted(particle_system.synset.children, key=lambda x: x.name):
                status_badge = self._status_to_badge(child.state)
                definition = getattr(child, 'definition', '') or '-'
                if len(definition) > 60:
                    definition = definition[:57] + "..."
                    
                categories_count = len(child.categories) if hasattr(child, 'categories') and child.categories else 0
                objects_count = 0
                if hasattr(child, 'matching_objects') and child.matching_objects:
                    objects_count = len(child.matching_objects)
                elif hasattr(child, 'categories') and child.categories:
                    for cat in child.categories:
                        if hasattr(cat, 'objects') and cat.objects:
                            objects_count += len(cat.objects)
                    
                content += f"| [{child.name}](../synsets/{child.name}.md) | {status_badge} | {definition} | {categories_count} | {objects_count} |\n"
            content += "\n"

        # Synset Graph Visualization
        if hasattr(particle_system, 'synset') and particle_system.synset:
            content += "## Related Synsets Graph\n\n"
            content += "```mermaid\n"
            content += "graph TD\n"
            
            # Add the main synset node
            main_synset = particle_system.synset
            content += f'    {main_synset.name.replace(" ", "_")}["{main_synset.name}"]\n'
            content += f'    {main_synset.name.replace(" ", "_")} --> {particle_system.name.replace(" ", "_")}["{particle_system.name}"]\n'
            
            # Add parent relationships
            if hasattr(main_synset, 'parents') and main_synset.parents:
                for parent in main_synset.parents[:5]:  # Limit to 5 parents
                    parent_id = parent.name.replace(" ", "_")
                    content += f'    {parent_id}["{parent.name}"] --> {main_synset.name.replace(" ", "_")}\n'
            
            # Add child relationships  
            if hasattr(main_synset, 'children') and main_synset.children:
                for child in main_synset.children[:5]:  # Limit to 5 children
                    child_id = child.name.replace(" ", "_")
                    content += f'    {main_synset.name.replace(" ", "_")} --> {child_id}["{child.name}"]\n'
            
            content += "```\n\n"
            
        # Ancestor Synsets Table
        ancestors = []
        if hasattr(particle_system, 'synset') and particle_system.synset:
            if hasattr(particle_system.synset, 'ancestors'):
                ancestors = list(particle_system.synset.ancestors)
            elif hasattr(particle_system.synset, 'parents'):
                # Build ancestors recursively
                visited = set()
                def get_ancestors(syn):
                    if syn in visited:
                        return []
                    visited.add(syn)
                    result = []
                    if hasattr(syn, 'parents') and syn.parents:
                        for parent in syn.parents:
                            result.append(parent)
                            result.extend(get_ancestors(parent))
                    return result
                ancestors = get_ancestors(particle_system.synset)
                
        if ancestors:
            content += "## Ancestor Synsets\n\n"
            content += "| Name | Status | Definition |\n"
            content += "|------|--------|------------|\n"
            for ancestor in sorted(set(ancestors), key=lambda x: x.name):
                status_badge = self._status_to_badge(ancestor.state)
                definition = getattr(ancestor, 'definition', '') or '-'
                if len(definition) > 80:
                    definition = definition[:77] + "..."
                content += f"| [{ancestor.name}](../synsets/{ancestor.name}.md) | {status_badge} | {definition} |\n"
            content += "\n"
        
        # Related Synsets
        related_synsets = set()
        if hasattr(particle_system, 'synset') and particle_system.synset:
            if hasattr(particle_system.synset, 'related_synsets'):
                related_synsets.update(particle_system.synset.related_synsets)
            # Add synsets that share similar particle properties
            if hasattr(particle_system.synset, 'categories') and particle_system.synset.categories:
                for cat in particle_system.synset.categories:
                    if hasattr(cat, 'synset') and cat.synset and cat.synset != particle_system.synset:
                        related_synsets.add(cat.synset)
                        
        if related_synsets:
            content += "## Related Synsets\n\n"
            content += "| Name | Status | Relationship | Common Properties |\n"
            content += "|------|--------|--------------|-------------------|\n"
            for related in sorted(related_synsets, key=lambda x: x.name):
                status_badge = self._status_to_badge(related.state)
                
                # Determine relationship
                relationship = "Related"
                if hasattr(particle_system, 'synset') and particle_system.synset:
                    if hasattr(particle_system.synset, 'parents') and related in (particle_system.synset.parents or []):
                        relationship = "Parent"
                    elif hasattr(particle_system.synset, 'children') and related in (particle_system.synset.children or []):
                        relationship = "Child"
                        
                # Find common properties (simplified)
                common_props = "-"
                if hasattr(related, 'system_type') and hasattr(particle_system, 'system_type'):
                    if getattr(related, 'system_type', None) == getattr(particle_system, 'system_type', None):
                        common_props = "system_type"
                    
                content += f"| [{related.name}](../synsets/{related.name}.md) | {status_badge} | {relationship} | {common_props} |\n"
            content += "\n"
        
        # Related objects
        if hasattr(particle_system, 'objects') and particle_system.objects:
            content += "## Objects\n\n"
            content += "| Object | Category | Ready | Provider |\n"
            content += "|--------|----------|-------|----------|\n"
            
            for obj in sorted(particle_system.objects, key=lambda x: x.name):
                # Category
                category_name = "-"
                if hasattr(obj, 'category') and obj.category:
                    category_name = f"[{obj.category.name}](../categories/{obj.category.name}.md)"
                elif hasattr(obj, 'owner') and obj.owner:
                    category_name = f"[{obj.owner.name}](../categories/{obj.owner.name}.md)"
                
                # Ready status
                ready_status = '✅' if getattr(obj, 'ready', False) else '⚠️'
                
                # Provider
                provider = getattr(obj, 'provider', '-')
                
                content += f"| [{obj.name}](../objects/{obj.name}.md) | {category_name} | {ready_status} | {provider} |\n"
            content += "\n"
            
        return content
        
    def _generate_task_table(self, tasks):
        """Generate comprehensive task list table."""
        # Use pure HTML table for better control
        content = """<style>
.tasks-table {
    font-size: 0.8em;
    width: 100%;
    table-layout: fixed;
    border-collapse: collapse;
    overflow-wrap: break-word;
    margin: 1em 0;
}
.tasks-table th {
    background-color: #f5f5f5;
    border: 1px solid #ddd;
    padding: 12px 8px;
    text-align: left;
    font-weight: bold;
    color: #333;
}
.tasks-table th:nth-child(1) { width: 18%; }
.tasks-table th:nth-child(2) { width: 28%; }
.tasks-table th:nth-child(3) { width: 22%; }
.tasks-table th:nth-child(4) { width: 18%; }
.tasks-table th:nth-child(5) { width: 14%; }
.tasks-table td {
    border: 1px solid #ddd;
    vertical-align: top;
    padding: 10px 8px;
    line-height: 1.4;
}
.tasks-table tr:nth-child(even) {
    background-color: #fafafa;
}
.tasks-table tr:hover {
    background-color: #f0f0f0;
}
.tasks-table a {
    color: #1976d2;
    text-decoration: none;
    font-size: 0.9em;
    display: inline-block;
    margin: 1px 0;
}
.tasks-table a:hover {
    color: #0d47a1;
    text-decoration: underline;
}
.tasks-table .status-badge {
    font-size: 0.75em;
    margin-top: 3px;
    display: inline-block;
    color: #666;
}
.tasks-table .status-icon {
    margin-right: 3px;
    font-size: 0.9em;
}
.tasks-table .task-name {
    font-weight: bold;
    color: #333;
}
.tasks-table .predicate-item, 
.tasks-table .feature-item {
    display: inline-block;
    margin: 1px 0;
    color: #555;
}
</style>

<table class="tasks-table">
<thead>
<tr>
<th>Task</th>
<th>Synsets</th>
<th>Matched Scenes</th>
<th>Predicates</th>
<th>Features</th>
</tr>
</thead>
<tbody>"""
        
        for task in sorted(tasks, key=lambda x: x.name):
            # Task name with status
            synset_badge = self._status_to_badge(task.synset_state)
            task_cell = f'<a href="{task.name}.md" class="task-name">{task.name}</a><br/><span class="status-badge">{synset_badge}</span>'
            
            # Synsets column - show ALL synsets with proper spacing
            synsets_list = []
            if task.synsets:
                for synset in sorted(task.synsets, key=lambda x: x.name):
                    status_icon = "✅" if synset.state.name == "MATCHED" else "📋" if synset.state.name == "PLANNED" else "❌"
                    synsets_list.append(f'<span class="status-icon">{status_icon}</span><a href="../synsets/{synset.name}.md">{synset.name}</a>')
                synsets_cell = "<br/>".join(synsets_list)
            else:
                synsets_cell = ""
            
            # Matched scenes column - show ALL matched scenes
            matched_scenes = []
            if hasattr(task, 'scene_matching_dict') and task.scene_matching_dict:
                for scene, info in sorted(task.scene_matching_dict.items(), key=lambda x: x[0].name):
                    if isinstance(info, dict):
                        matched = info.get('matched', False)
                    else:
                        matched = getattr(info, 'matched', False)
                    
                    if matched:
                        matched_scenes.append(f'<a href="../scenes/{scene.name}.md">{scene.name}</a>')
                
                scenes_cell = "<br/>".join(matched_scenes)
            else:
                scenes_cell = ""
            
            # Predicates column - show ALL predicates with better formatting
            if hasattr(task, 'uses_predicates') and task.uses_predicates:
                predicates_list = [f'<span class="predicate-item">{pred.name}</span>' for pred in task.uses_predicates]
                predicates_cell = "<br/>".join(predicates_list)
            else:
                predicates_cell = ""
            
            # Required Features column with better formatting
            features = []
            if hasattr(task, 'uses_transition') and task.uses_transition:
                features.append('<span class="feature-item">transition</span>')
            if hasattr(task, 'uses_visual_substance') and task.uses_visual_substance:
                features.append('<span class="feature-item">visual substance</span>')
            if hasattr(task, 'uses_physical_substance') and task.uses_physical_substance:
                features.append('<span class="feature-item">physical substance</span>')
            if hasattr(task, 'uses_attachment') and task.uses_attachment:
                features.append('<span class="feature-item">attachment</span>')
            if hasattr(task, 'uses_cloth') and task.uses_cloth:
                features.append('<span class="feature-item">cloth</span>')
            
            features_cell = "<br/>".join(features)
            
            content += f'\n<tr><td>{task_cell}</td><td>{synsets_cell}</td><td>{scenes_cell}</td><td>{predicates_cell}</td><td>{features_cell}</td></tr>'
        
        content += "\n</tbody>\n</table>"
        return content
        
    def _generate_task_table_simple(self, tasks):
        """Generate simple Markdown task table with complete information."""
        content = "| Task | Synsets | Matched Scenes | Predicates | Required Features |\n"
        content += "|------|---------|----------------|------------|----------|\n"
        
        for task in sorted(tasks, key=lambda x: x.name):
            # Task name
            task_name = f"[{task.name}]({task.name}.md)"
            
            # Synsets - show ALL synsets with status icons and links
            synsets_list = []
            if task.synsets:
                for synset in sorted(task.synsets, key=lambda x: x.name):
                    status_icon = "✅" if synset.state.name == "MATCHED" else "📋" if synset.state.name == "PLANNED" else "❌"
                    synsets_list.append(f'{status_icon} [{synset.name}](../synsets/{synset.name}.md)')
                synsets_cell = "<br/>".join(synsets_list)
            else:
                synsets_cell = "-"
            
            # Matched scenes - show ALL matched scene names with links
            matched_scenes = []
            if hasattr(task, 'scene_matching_dict') and task.scene_matching_dict:
                for scene, info in sorted(task.scene_matching_dict.items(), key=lambda x: x[0].name):
                    if isinstance(info, dict):
                        matched = info.get('matched', False)
                    else:
                        matched = getattr(info, 'matched', False)
                    
                    if matched:
                        matched_scenes.append(f'[{scene.name}](../scenes/{scene.name}.md)')
                
                scenes_cell = "<br/>".join(matched_scenes) if matched_scenes else "-"
            else:
                scenes_cell = "-"
            
            # Predicates - show ALL predicates used by the task
            if hasattr(task, 'uses_predicates') and task.uses_predicates:
                predicates_list = [pred.name for pred in task.uses_predicates]
                predicates_cell = "<br/>".join(predicates_list)
            else:
                predicates_cell = "-"
            
            # Features
            features = []
            if hasattr(task, 'uses_transition') and task.uses_transition:
                features.append('transition')
            if hasattr(task, 'uses_visual_substance') and task.uses_visual_substance:
                features.append('visual substance')
            if hasattr(task, 'uses_physical_substance') and task.uses_physical_substance:
                features.append('physical substance')
            if hasattr(task, 'uses_attachment') and task.uses_attachment:
                features.append('attachment')
            if hasattr(task, 'uses_cloth') and task.uses_cloth:
                features.append('cloth')
            
            features_str = "<br/>".join(features) if features else "-"
            
            content += f"| {task_name} | {synsets_cell} | {scenes_cell} | {predicates_cell} | {features_str} |\n"
        
        return content
        
    def _generate_synset_table(self, synsets):
        """Generate simple Markdown synset table with complete information."""
        content = "| Name | Status | Custom | Leaf | Definition | Parents | Children | Properties | Predicates | Tasks | Categories | Direct Obj | Total Obj | Task Count |\n"
        content += "|------|--------|--------|------|------------|---------|----------|------------|------------|-------|------------|------------|-----------|------------|\n"
        
        for synset in sorted(synsets, key=lambda x: x.name):
            # Name
            name_cell = f"[{synset.name}]({synset.name}.md)"
            
            # Status
            state_badge = self._status_to_badge(synset.state)
            
            # Custom (check if it's a custom synset)
            is_custom = getattr(synset, 'is_custom', False)
            custom_cell = "Yes" if is_custom else "No"
            
            # Leaf (True if no children)
            is_leaf = len(synset.children) == 0 if synset.children else True
            leaf_cell = "Yes" if is_leaf else "No"
            
            # Definition
            definition = getattr(synset, 'definition', '') or ''
            if len(definition) > 50:
                definition = definition[:47] + "..."
            definition_cell = definition if definition else "-"
            
            # Parents
            parents_list = []
            if synset.parents:
                for parent in synset.parents:
                    parents_list.append(f'[{parent.name}]({parent.name}.md)')
            parents_cell = "<br/>".join(parents_list) if parents_list else "-"
            
            # Children
            children_list = []
            if synset.children:
                for child in sorted(synset.children, key=lambda x: x.name):
                    children_list.append(f'[{child.name}]({child.name}.md)')
            if len(children_list) > 5:
                children_cell = "<br/>".join(children_list[:5]) + f"<br/>...+{len(children_list)-5} more"
            else:
                children_cell = "<br/>".join(children_list) if children_list else "-"
            
            # Properties
            properties_list = []
            if hasattr(synset, 'property_names') and synset.property_names:
                properties_list = list(synset.property_names)
            elif hasattr(synset, 'positive_properties') and synset.positive_properties:
                properties_list = [prop for prop in synset.positive_properties]
            properties_cell = "<br/>".join(properties_list) if properties_list else "-"
            
            # Used as Predicates
            predicates_list = []
            if hasattr(synset, 'used_in_predicates') and synset.used_in_predicates:
                predicates_list = [pred.name for pred in synset.used_in_predicates]
            if len(predicates_list) > 5:
                predicates_cell = "<br/>".join(predicates_list[:5]) + f"<br/>...+{len(predicates_list)-5} more"
            else:
                predicates_cell = "<br/>".join(predicates_list) if predicates_list else "-"
            
            # Tasks
            tasks_list = []
            if hasattr(synset, 'tasks') and synset.tasks:
                for task in synset.tasks[:5]:  # Limit to first 5
                    tasks_list.append(f'[{task.name}](../tasks/{task.name}.md)')
                if len(synset.tasks) > 5:
                    tasks_cell = "<br/>".join(tasks_list) + f"<br/>...+{len(synset.tasks)-5} more"
                else:
                    tasks_cell = "<br/>".join(tasks_list)
            else:
                tasks_cell = "-"
            
            # Direct Categories
            direct_cat_count = len(synset.categories) if synset.categories else 0
            
            # Direct Objects
            direct_obj_count = 0
            if hasattr(synset, 'direct_matching_objects'):
                direct_obj_count = len(synset.direct_matching_objects)
            
            # Total Objects
            total_obj_count = 0
            if hasattr(synset, 'matching_objects'):
                total_obj_count = len(synset.matching_objects)
            
            # Requiring Task Count
            task_count = 0
            if hasattr(synset, 'n_task_required'):
                task_count = synset.n_task_required
            elif hasattr(synset, 'tasks'):
                task_count = len(synset.tasks)
            
            content += f"| {name_cell} | {state_badge} | {custom_cell} | {leaf_cell} | {definition_cell} | {parents_cell} | {children_cell} | {properties_cell} | {predicates_cell} | {tasks_cell} | {direct_cat_count} | {direct_obj_count} | {total_obj_count} | {task_count} |\n"
        
        return content
        
    def _generate_object_table(self, objects):
        """Generate simple Markdown object table with complete information."""
        content = "| Object | Category/System | Synset | Bounding Box | Meta Links | Used Rooms | Ready |\n"
        content += "|--------|-----------------|--------|--------------|------------|------------|-------|\n"
        
        for obj in sorted(objects, key=lambda x: x.name):
            # Object name
            name_cell = f"[{obj.name}]({obj.name}.md)"
            
            # Category or Particle System
            if obj.category:
                owner_cell = f'[{obj.category.name}](../categories/{obj.category.name}.md)'
            elif obj.particle_system:
                owner_cell = f'[{obj.particle_system.name}](../particle_systems/{obj.particle_system.name}.md)'
            else:
                owner_cell = "-"
            
            # Synset
            if hasattr(obj, 'owner') and obj.owner and hasattr(obj.owner, 'synset') and obj.owner.synset:
                synset_cell = f'[{obj.owner.synset.name}](../synsets/{obj.owner.synset.name}.md)'
            elif obj.category and hasattr(obj.category, 'synset') and obj.category.synset:
                synset_cell = f'[{obj.category.synset.name}](../synsets/{obj.category.synset.name}.md)'
            elif obj.particle_system and hasattr(obj.particle_system, 'synset') and obj.particle_system.synset:
                synset_cell = f'[{obj.particle_system.synset.name}](../synsets/{obj.particle_system.synset.name}.md)'
            else:
                synset_cell = "-"
            
            # Bounding Box Size
            if obj.bounding_box_size and len(obj.bounding_box_size) >= 3:
                bbox_cell = f'{obj.bounding_box_size[0]:.2f}×{obj.bounding_box_size[1]:.2f}×{obj.bounding_box_size[2]:.2f}'
            else:
                bbox_cell = "-"
            
            # Meta Links
            meta_links_list = []
            if hasattr(obj, 'meta_links') and obj.meta_links:
                for ml in obj.meta_links:
                    if hasattr(ml, 'name'):
                        meta_links_list.append(ml.name)
                    else:
                        meta_links_list.append(str(ml))
                if len(meta_links_list) > 3:
                    meta_links_cell = "<br/>".join(meta_links_list[:3]) + f"<br/>...+{len(meta_links_list)-3} more"
                else:
                    meta_links_cell = "<br/>".join(meta_links_list)
            else:
                meta_links_cell = "-"
            
            # Used Rooms
            if hasattr(obj, 'roomobjects') and obj.roomobjects:
                rooms_count = len(obj.roomobjects)
            else:
                rooms_count = 0
            
            # Ready Status
            ready_status = getattr(obj, 'ready', False)
            if ready_status:
                ready_cell = '✅ Ready'
            else:
                ready_cell = '⚠️ Review'
            
            content += f"| {name_cell} | {owner_cell} | {synset_cell} | {bbox_cell} | {meta_links_cell} | {rooms_count} | {ready_cell} |\n"
        
        return content
        
    def _generate_scene_table(self, scenes):
        """Generate scene list table."""
        content = "| Scene | Rooms | Objects |\n"
        content += "|-------|-------|---------|\n"
        
        for scene in sorted(scenes, key=lambda x: x.name):
            # Use room_count property if available, otherwise count rooms
            if hasattr(scene, 'room_count'):
                room_count = scene.room_count
            else:
                room_count = len(scene.rooms) if hasattr(scene, 'rooms') and scene.rooms else 0
                
            # Use object_count property if available (as in original template)
            if hasattr(scene, 'object_count'):
                object_count = scene.object_count
            elif hasattr(scene, 'objects') and scene.objects:
                object_count = len(scene.objects)
            elif hasattr(scene, 'room_objects') and scene.room_objects:
                object_count = len(scene.room_objects)
            else:
                object_count = 0
                
            content += f"| [{scene.name}]({scene.name}.md) | {room_count} | {object_count} |\n"
            
        return content
        
    def _generate_category_table(self, categories):
        """Generate category list table."""
        content = "| Category | Synset | Objects | Ready Objects |\n"
        content += "|----------|--------|---------|---------------|\n"
        
        for cat in sorted(categories, key=lambda x: x.name):
            # Fix synset linking
            if cat.synset:
                synset_name = f"[{cat.synset.name}](../synsets/{cat.synset.name}.md)"
            else:
                synset_name = "-"
            
            obj_count = len(cat.objects) if cat.objects else 0
            ready_count = sum(1 for o in cat.objects if getattr(o, 'ready', False)) if cat.objects else 0
            content += f"| [{cat.name}]({cat.name}.md) | {synset_name} | {obj_count} | {ready_count} |\n"
            
        return content
        
    def _generate_transition_table(self, rules):
        """Generate transition rule list table."""
        content = "| Rule | Input Synsets | Machine Synsets | Output Synsets |\n"
        content += "|------|---------------|-----------------|----------------|\n"
        
        for rule in sorted(rules, key=lambda x: x.name):
            # Input synsets - show actual synset names
            input_synsets = []
            if hasattr(rule, 'input_synsets') and rule.input_synsets:
                input_synsets = [f"[{syn.name}](../synsets/{syn.name}.md)" for syn in rule.input_synsets[:3]]
                if len(rule.input_synsets) > 3:
                    input_synsets.append(f"...+{len(rule.input_synsets)-3}")
            input_cell = "<br/>".join(input_synsets) if input_synsets else "-"
            
            # Machine synsets - show actual synset names  
            machine_synsets = []
            if hasattr(rule, 'machine_synsets') and rule.machine_synsets:
                machine_synsets = [f"[{syn.name}](../synsets/{syn.name}.md)" for syn in rule.machine_synsets[:3]]
                if len(rule.machine_synsets) > 3:
                    machine_synsets.append(f"...+{len(rule.machine_synsets)-3}")
            elif hasattr(rule, 'tool_synsets') and rule.tool_synsets:
                machine_synsets = [f"[{syn.name}](../synsets/{syn.name}.md)" for syn in rule.tool_synsets[:3]]
                if len(rule.tool_synsets) > 3:
                    machine_synsets.append(f"...+{len(rule.tool_synsets)-3}")
            machine_cell = "<br/>".join(machine_synsets) if machine_synsets else "-"
            
            # Output synsets - show actual synset names
            output_synsets = []
            if hasattr(rule, 'output_synsets') and rule.output_synsets:
                output_synsets = [f"[{syn.name}](../synsets/{syn.name}.md)" for syn in rule.output_synsets[:3]]
                if len(rule.output_synsets) > 3:
                    output_synsets.append(f"...+{len(rule.output_synsets)-3}")
            output_cell = "<br/>".join(output_synsets) if output_synsets else "-"
            
            content += f"| [{rule.name}]({rule.name}.md) | {input_cell} | {machine_cell} | {output_cell} |\n"
            
        return content
        
    def _generate_attachment_table(self, pairs):
        """Generate attachment pair list table."""
        content = "| Name | Female Objects Count | Male Objects Count |\n"
        content += "|------|----------------------|--------------------|\n"
        
        for pair in sorted(pairs, key=lambda x: x.name):
            # Count female objects
            female_count = 0
            if hasattr(pair, 'female_objects') and pair.female_objects:
                female_count = len(pair.female_objects)
            elif hasattr(pair, 'female_synset') and pair.female_synset:
                # If we have a female synset, we can try to get its object count
                if hasattr(pair.female_synset, 'matching_objects'):
                    female_count = len(pair.female_synset.matching_objects)
                elif hasattr(pair.female_synset, 'objects'):
                    female_count = len(pair.female_synset.objects)
                    
            # Count male objects
            male_count = 0
            if hasattr(pair, 'male_objects') and pair.male_objects:
                male_count = len(pair.male_objects)
            elif hasattr(pair, 'male_synset') and pair.male_synset:
                # If we have a male synset, we can try to get its object count
                if hasattr(pair.male_synset, 'matching_objects'):
                    male_count = len(pair.male_synset.matching_objects)
                elif hasattr(pair.male_synset, 'objects'):
                    male_count = len(pair.male_synset.objects)
                    
            content += f"| [{pair.name}]({pair.name}.md) | {female_count} | {male_count} |\n"
            
        return content
        
    def _generate_particle_table(self, particles):
        """Generate particle system list table."""
        content = "| Particle System | Synset |\n"
        content += "|-----------------|--------|\n"
        
        for particle in sorted(particles, key=lambda x: x.name):
            synset_name = particle.synset.name if particle.synset else "-"
            content += f"| [{particle.name}]({particle.name}.md) | {synset_name} |\n"
            
        return content
        
    def _generate_complaint_type_table(self, complaints):
        """Generate complaint type list table without links to detail pages."""
        content = "| Complaint Type | Affected Objects |\n"
        content += "|----------------|------------------|\n"
        
        for complaint in sorted(complaints, key=lambda x: x.name):
            obj_count = len(complaint.objects) if hasattr(complaint, 'objects') else 0
            # Note: No link to detail page, just display the name
            content += f"| {complaint.name} | {obj_count} |\n"
            
        return content
        
    def _generate_mermaid_graph(self, graph, center_node=None):
        """Generate Mermaid diagram for graph."""
        content = "```mermaid\ngraph TD\n"
        
        # Add edges
        for edge in graph.edges:
            parent = edge[0].name if hasattr(edge[0], 'name') else str(edge[0])
            child = edge[1].name if hasattr(edge[1], 'name') else str(edge[1])
            content += f"    {self._sanitize_mermaid_id(parent)} --> {self._sanitize_mermaid_id(child)}\n"
            
        # Add node styling
        for node in graph.nodes:
            node_name = node.name if hasattr(node, 'name') else str(node)
            node_id = self._sanitize_mermaid_id(node_name)
            
            # Apply color based on state
            if hasattr(node, 'state'):
                color_class = self._state_to_mermaid_class(node.state)
                content += f"    class {node_id} {color_class}\n"
                
            # Highlight center node
            if center_node and node == center_node:
                content += f"    style {node_id} stroke:#333,stroke-width:3px\n"
                
        # Define color classes
        content += """
    classDef matched fill:#28a745,color:#fff
    classDef planned fill:#ffc107,color:#000
    classDef unmatched fill:#dc3545,color:#fff
    classDef illegal fill:#6c757d,color:#fff
"""
        content += "```\n"
        
        return content
        
    def generate_nav_file(self):
        """Generate a navigation structure file for MkDocs."""
        nav_structure = {
            "Knowledgebase": {
                "Overview": "knowledgebase/index.md",
                "Tasks": {
                    "All Tasks": "knowledgebase/tasks/index.md",
                },
                "Synsets": "knowledgebase/synsets/index.md",
                "Objects": "knowledgebase/objects/index.md",
                "Categories": "knowledgebase/categories/index.md",
                "Scenes": "knowledgebase/scenes/index.md",
                "Transition Rules": "knowledgebase/transition_rules/index.md",
                "Particle Systems": "knowledgebase/particle_systems/index.md",
                "Attachment Pairs": "knowledgebase/attachment_pairs/index.md"
            }
        }
        
        # Save as JSON for potential use
        nav_path = self.output_dir / "nav.json"
        nav_path.write_text(json.dumps(nav_structure, indent=2))
        
    def _snake_case(self, name):
        """Convert CamelCase to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        
    def _pluralize(self, word):
        """Simple pluralization."""
        if word.endswith('y'):
            return word[:-1] + 'ies'
        elif word.endswith('s') or word.endswith('x'):
            return word + 'es'
        else:
            return word + 's'
            
    def _sanitize_mermaid_id(self, text):
        """Sanitize text for use as Mermaid node ID."""
        # Replace special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', text)
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'n_' + sanitized
        return sanitized
        
    def _status_to_badge(self, state):
        """Convert state to badge."""
        if state == SynsetState.MATCHED:
            return "✅ READY"
        elif state == SynsetState.PLANNED:
            return "📋 PLANNED"
        elif state == SynsetState.UNMATCHED:
            return "❌ UNMATCHED"
        elif state == SynsetState.ILLEGAL:
            return "⛔ ILLEGAL"
        else:
            return str(state)
            
    def _status_to_badge_color(self, state):
        """Convert state to color."""
        if state == SynsetState.MATCHED:
            return "green"
        elif state == SynsetState.PLANNED:
            return "orange"
        elif state == SynsetState.UNMATCHED:
            return "red"
        elif state == SynsetState.ILLEGAL:
            return "gray"
        else:
            return "black"
            
    def _state_to_mermaid_class(self, state):
        """Convert state to Mermaid class name."""
        if state == SynsetState.MATCHED:
            return "matched"
        elif state == SynsetState.PLANNED:
            return "planned"
        elif state == SynsetState.UNMATCHED:
            return "unmatched"
        elif state == SynsetState.ILLEGAL:
            return "illegal"
        else:
            return "default"


def main():
    parser = argparse.ArgumentParser(description="Generate Markdown files for BEHAVIOR Knowledgebase")
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('docs/knowledgebase'),
        help='Output directory for generated Markdown files'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean output directory before generating'
    )
    
    args = parser.parse_args()
    
    # Clean if requested
    if args.clean and args.output.exists():
        import shutil
        print(f"Cleaning {args.output}...")
        shutil.rmtree(args.output)
    
    # Generate files
    generator = MarkdownGenerator(args.output)
    files = generator.generate_all()
    
    print(f"\n✅ Generated {len(files)} Markdown files in {args.output}")
    print("\nTo integrate with MkDocs:")
    print("1. Update mkdocs.yml to include the knowledgebase section")
    print("2. Run: mkdocs serve")
    

if __name__ == "__main__":
    main()