# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the BEHAVIOR-1K Knowledgebase - a Flask web application that provides an interactive interface for browsing and searching the BEHAVIOR-1K dataset's knowledge base. It serves as a dashboard for exploring various BDDL (Behavior Domain Definition Language) entities including objects, scenes, tasks, and more.

## Commands

### Static Site Generation (Primary Method)
```bash
# Install dependencies for static generation
pip install -r requirements_static.txt

# Generate static site (default output: build/)
python static_generator.py

# Generate with custom output directory
python static_generator.py -o dist

# Generate with more parallel workers for faster generation
python static_generator.py -w 8

# Serve the static site locally for testing
python -m http.server 8000 --directory build
```

### Flask Development Server (Legacy)
```bash
# Install Flask dependencies
pip install -r requirements.txt

# Run the Flask development server
flask --app knowledgebase.app run

# Run in background
flask --app knowledgebase.app run &
```

### Development Commands
```bash
# Check Python syntax errors
python -m py_compile knowledgebase/*.py static_generator.py

# Install new dependencies
pip install <package>
# For static generator:
pip freeze > requirements_static.txt
# For Flask:
pip freeze > requirements.txt
```

## Architecture

### Core Structure
The application dynamically generates views for BDDL knowledge base models using Flask's class-based views:

- **Dynamic View Generation**: The app automatically creates ListView and DetailView classes for each model at runtime (see `knowledgebase/app.py:57-96`)
- **Models**: Imported from `bddl.knowledge_base` - includes Task, Scene, Object, Synset, Category, TransitionRule, AttachmentPair, ComplaintType, ParticleSystem
- **Template Inheritance**: All views inherit from base templates (`templates/base.html`) with model-specific templates following the pattern `{model}_list.html` and `{model}_detail.html`

### Key Components

1. **`knowledgebase/app.py`**: Main application with route registration and dynamic view creation
   - Dynamically creates routes like `/knowledgebase/{model_plural}/` for lists and `/knowledgebase/{model_plural}/{id}/` for details
   - Configures template auto-reload for development

2. **`knowledgebase/views.py`**: Base view classes (ListView, DetailView, IndexView)
   - ListView handles pagination and search
   - DetailView handles individual entity display
   - Contains `searchable_items_list` for generating JSON search index

3. **`knowledgebase/profile_utils.py`**: Performance monitoring utilities
   - Fetches and processes profiling data from external OmniGibson instances
   - Generates performance badges and plots

4. **`knowledgebase/filters.py`**: Template filters for status color mapping
   - Maps entity statuses (MATCHED, PLANNED, UNMATCHED, ILLEGAL) to Bootstrap colors

### Frontend Architecture
- **Search**: Client-side fuzzy search using Fuse.js (`templates/base.html`)
- **Visualizations**: Mermaid diagrams with D3.js zoom/pan functionality
- **UI Framework**: Bootstrap 4.3.1 with jQuery for interactivity

### Data Flow
1. Models are imported from the BDDL package
2. Views are dynamically created for each model type
3. Templates render data with status color coding and interactive features
4. Search functionality operates on client-side cached JSON data

## Development Notes

- **No test suite**: This is a research/exploration tool without automated tests
- **Template auto-reload**: Enabled by default for development (`app.config["TEMPLATES_AUTO_RELOAD"] = True`)
- **Debug mode**: Enabled by default (`app.debug = True`)
- **No linting configuration**: Project doesn't include flake8, black, or ruff configs
- **Simple deployment**: No Docker, CI/CD, or complex build process

## Key Files to Understand

- `knowledgebase/app.py`: Entry point and dynamic route generation
- `knowledgebase/views.py`: View logic and search index generation
- `templates/base.html`: Main template with navigation and search functionality
- `templates/index.html`: Dashboard with statistics and overview