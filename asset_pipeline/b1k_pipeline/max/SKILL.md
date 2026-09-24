# 3ds Max Live Scene Skill

Use this skill when you need to inspect or modify the currently open Autodesk 3ds Max scene with Python. 3ds Max exposes MAXScript through `pymxs`, and this repository already uses that pattern throughout `asset_pipeline/b1k_pipeline/max`.

## Required workflow

1. Run Python inside 3ds Max, not in the normal project interpreter. Start the MCP server from the 3ds Max Python interpreter:
   ```python
   exec(open(r"D:\BEHAVIOR-1K\asset_pipeline\b1k_pipeline\max\mcp_repl_server.py").read())
   ```
   Adjust the path if the checkout is elsewhere.
2. Connect your MCP client to that process over stdio. The server exposes two tools: `run_python` for inline snippets and `execute_script` for a Python script at a full path on disk.
3. 3ds Max uses Python 3.7 in this environment. All snippets and scripts sent through the MCP server must be Python 3.7-compatible: avoid newer syntax such as `X | Y` type unions, built-in generic annotations like `list[str]`, `match` statements, and other Python 3.8+ features.
4. Use `run_python` for exploratory live-scene work. The namespace is stateful and already contains:
   ```python
   import pymxs
   rt = pymxs.runtime
   ```
5. Use `execute_script` when code is long enough that it should live in a temporary or repository script file. Pass `script_path` as an absolute path; optionally pass `script_args` to populate `sys.argv[1:]`.
6. Prefer short read-only probes before edits. Tool responses include `stdout`, `stderr`, any error traceback, and a compact `scene_state` summary of objects, selection, cameras, lights, materials, and render settings. `run_python` also returns `result_repr` and `result_json` for the final expression. Pass `capture_view=True` to either tool to also save the active viewport to a PNG and return its path.
7. Make minimal, reversible changes. Save files only when explicitly requested or after confirming the desired result.

## Finding MAXScript / pymxs documentation

`pymxs` is a wrapper over MAXScript: most functions, globals, classes, and interfaces correspond 1:1 with MAXScript names. Autodesk's pymxs introduction says the module exposes virtually all MAXScript interfaces, globals, and classes. Autodesk forum guidance also recommends consulting the MAXScript Guide once you understand value translation.

Useful searches:

- `site:help.autodesk.com 3ds Max MAXScript polyOp getVert`
- `site:help.autodesk.com 3ds Max MAXScript viewport setLayout setCamera`
- `site:help.autodesk.com 3ds Max MAXScript render outputFile`
- `site:help.autodesk.com 3ds Max MAXScript VRay renderers current`
- `site:help.autodesk.com 3ds Max pymxs byref Name Array BitArray`

When translating docs, keep the MAXScript call shape and use Python syntax:

| MAXScript | pymxs Python |
| --- | --- |
| `$.name` | `rt.selection[0].name` |
| `$Box001.material` | `rt.getNodeByName("Box001").material` |
| `classOf obj` | `rt.classOf(obj)` |
| `polyOp.getNumVerts obj` | `rt.polyop.getNumVerts(obj)` |
| `polyOp.getVert obj 1` | `rt.polyop.getVert(obj, 1)` |
| `#{1..10}` | `rt.execute("#{1..10}")` or a Python list when accepted |
| `#noPrompt` | `rt.name("noPrompt")` or `rt.Name("noPrompt")` |
| `render outputFile:path` | `rt.render(outputFile=path)` |
| `mergeMAXFile path #select` | `rt.mergeMAXFile(path, rt.name("select"))` |
| by-reference args | `ref = pymxs.byref(None); rt.someFn(arg, out=ref); ref.value` |

Notes:

- MAXScript arrays and bitarrays are often 1-indexed. `polyop` face and vertex indices are 1-indexed.
- Property access is usually direct (`obj.isHidden = True`, `mat.diffuse = rt.Color(255, 0, 0)`). Use `rt.isProperty(x, "propName")` before optional/plugin-specific properties.
- Use `rt.execute("...")` as an escape hatch for tricky MAXScript literals or contexts.
- The repository examples show common patterns: `rt.render(outputFile=...)`, `rt.viewport.setLayout(rt.Name("layout_1"))`, `rt.viewport.setCamera(camera)`, `rt.polyop.getVert`, `rt.polyop.getFacesVerts`, `rt.polyop.createVert`, `rt.polyop.createPolygon`, `rt.classOf(obj)`, `pymxs.byref(None)`, and VRay material constructors such as `rt.VRayMtl()`.

## Scene inspection examples

Run these snippets with the MCP `run_python` tool. Keep them Python 3.7-compatible.


### Execute a script from disk

Use the MCP `execute_script` tool for larger workflows that are easier to author as a file. The script runs inside the same stateful 3ds Max namespace as `run_python`, with `pymxs` and `rt` available if imported or referenced from globals. The tool captures stdout/stderr and returns scene state after execution.

```json
{
  "script_path": "C:\\temp\\inspect_scene.py",
  "script_args": ["--limit", "25"],
  "capture_view": false
}
```

Example `inspect_scene.py` content, written for Python 3.7:

```python
print("Scene:", rt.maxFilePath, rt.maxFileName)
print("Objects:", len(list(rt.objects)))
for obj in list(rt.objects)[:10]:
    print(obj.name, rt.classOf(obj), getattr(obj, "isHidden", False))
```

### List objects and selected nodes

```python
[(obj.name, str(rt.classOf(obj)), obj.isHidden) for obj in rt.objects]
```

```python
[obj.name for obj in rt.selection]
```

### Read Editable Poly geometry

```python
obj = rt.selection[0]
num_verts = rt.polyop.getNumVerts(obj)
num_faces = rt.polyop.getNumFaces(obj)
verts = [tuple(rt.polyop.getVert(obj, i + 1)) for i in range(num_verts)]
faces = [list(rt.polyop.getFaceVerts(obj, i + 1)) for i in range(num_faces)]
{"name": obj.name, "verts": verts[:10], "faces": faces[:10], "counts": (num_verts, num_faces)}
```

To read all face vertices at once, mirror the pipeline scripts:

```python
obj = rt.selection[0]
face_bitarray = rt.execute("#{1..%d}" % rt.polyop.getNumFaces(obj))
faces_zero_based = [[idx - 1 for idx in face] for face in rt.polyop.getFacesVerts(obj, face_bitarray)]
faces_zero_based[:10]
```

### Inspect materials recursively

```python
def summarize_material(mat, depth=0):
    if mat is None:
        return None
    out = {"name": str(getattr(mat, "name", "")), "class": str(rt.classOf(mat))}
    for prop in ["diffuse", "diffuseColor", "reflection", "refraction", "texmap_diffuse", "fileName"]:
        if rt.isProperty(mat, prop):
            out[prop] = repr(getattr(mat, prop))
    if depth < 2 and rt.isProperty(mat, "materialList"):
        out["submaterials"] = [summarize_material(m, depth + 1) for m in mat.materialList if m]
    return out

[(obj.name, summarize_material(obj.material)) for obj in rt.objects if getattr(obj, "material", None)][:5]
```

### Modify or assign a VRay material

```python
obj = rt.selection[0]
mat = rt.VRayMtl()
mat.name = obj.name + "_agent_preview_vray"
if rt.isProperty(mat, "diffuse"):
    mat.diffuse = rt.Color(180, 180, 180)
obj.material = mat
obj.material.name
```

### Read and modify renderer / VRay settings

```python
{
    "renderer": str(rt.classOf(rt.renderers.current)),
    "width": rt.rendWidth,
    "height": rt.rendHeight,
    "output": rt.rendOutputFilename,
}
```

```python
rt.rendWidth = 800
rt.rendHeight = 600
if "VRay" not in str(rt.classOf(rt.renderers.current)):
    rt.renderers.current = rt.VRay()
str(rt.classOf(rt.renderers.current))
```

Plugin-specific VRay properties vary by version. Probe with `rt.getPropNames(rt.renderers.current)` and set only properties that exist:

```python
renderer = rt.renderers.current
props = [str(p) for p in rt.getPropNames(renderer)]
[p for p in props if "gi" in p.lower() or "image" in p.lower()][:50]
```

### Render a preview image

```python
import os, tempfile
path = os.path.join(tempfile.gettempdir(), "agent_render.png")
rt.rendWidth = 640
rt.rendHeight = 480
rt.render(outputFile=path)
path
```

If the scene has named cameras such as `camera-diag`, use the viewport/camera workflow from this repository:

```python
camera = rt.getNodeByName("camera-diag")
rt.viewport.setLayout(rt.Name("layout_1"))
rt.viewport.setCamera(camera)
rt.render(outputFile=r"C:\temp\camera-diag.png")
```

### Capture the active viewport instead of a full render

Use the MCP tool with `capture_view=True`, or run directly:

```python
import os, tempfile
path = os.path.join(tempfile.gettempdir(), "agent_viewport.png")
bitmap = rt.gw.getViewportDib()
rt.save(bitmap, path)
rt.close(bitmap)
path
```

For a repeatable four-angle viewport, create or select cameras named `camera-top`, `camera-front`, `camera-side`, and `camera-diag`, then switch and capture/render each one. If no cameras exist, start with built-in view types through MAXScript:

```python
rt.viewport.setLayout(rt.Name("layout_4"))
rt.execute("viewport.setType #view_top")
```

### Isolate one object, inspect it, then restore visibility

```python
# Save visibility by name so the operation is reversible.
_visibility = {obj.name: obj.isHidden for obj in rt.objects}
target = rt.selection[0]
for obj in rt.objects:
    obj.isHidden = obj != target
rt.select(target)
rt.redrawViews()
target.name
```

Restore later in the same stateful MCP session:

```python
for obj in rt.objects:
    if obj.name in _visibility:
        obj.isHidden = _visibility[obj.name]
rt.redrawViews()
"restored"
```

### Create or edit poly data

```python
mesh = rt.Editable_Poly(name="agent_triangle")
rt.polyop.createVert(mesh, rt.Point3(0, 0, 0))
rt.polyop.createVert(mesh, rt.Point3(10, 0, 0))
rt.polyop.createVert(mesh, rt.Point3(0, 10, 0))
rt.polyop.createPolygon(mesh, [1, 2, 3])
rt.update(mesh)
mesh.name
```

## Safety checklist before edits

- Capture current selection and visibility if you will hide/isolate objects.
- Read `rt.maxFilePath` and `rt.maxFileName`; do not overwrite files unexpectedly.
- Query `rt.classOf(obj)` before using `polyop` methods.
- For destructive geometry edits, duplicate first: `copy = rt.copy(obj); copy.name = obj.name + "_agent_copy"`.
- After edits, run a viewport capture or a small render and inspect the returned image path.
