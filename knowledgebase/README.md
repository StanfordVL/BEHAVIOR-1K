# Knowledgebase
A website view of the BEHAVIOR-1K knowledgebase included in the BDDL repository.

## Static Site Generation (Recommended)

Generate the entire website as static HTML files without running a server:

```bash
# Install requirements
pip install -r requirements_static.txt

# Generate static site
python static_generator.py -o build

# Optionally specify number of parallel workers (default: 4)
python static_generator.py -o build -w 8
```

The static files will be generated in the `build/` directory. You can then serve them with any static file server.

## Flask Development Server (Legacy)

To run with the Flask development server:

```bash
# Install requirements
pip install -r requirements.txt

# Run flask
flask --app knowledgebase.app run &
```