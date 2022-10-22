## Guide on Converting Matterport (or other) scans to OmniGibson format

This document contains steps to convert a matterport scan (it might work with other scan formats, as long as the data is represented as a 
triangular mesh) to OmniGibson format.

A typical matterport scan in "matterpak" format has the following file format:
```
├── <UUID>_000.jpg
├── <UUID>_001.jpg
├── <UUID>_002.jpg
├── <UUID>_003.jpg
├── <UUID>_004.jpg
├── <UUID>_005.jpg
├── <UUID>_006.jpg
├── <UUID>_007.jpg
├── <UUID>_008.jpg
├── <UUID>_009.jpg
├── <UUID>.mtl
├── <UUID>.obj
├── ceilingcolorplan_000.jpg
├── ceilingcolorplan.pdf
├── cloud.xyz
├── colorplan_000.jpg
├── colorplan.pdf
```
There are a few steps involved in converting this format to OmniGibson format:
1. Make sure it is actually in z-up convention. Usually, matterport meshes are already
following this convention. 
2. Combine all the textures into one file, and modify the `mtl` file and `obj` file accordingly. It can be done with `combine_texture.py`, please follow the steps there. 
3. (required) Add surface normals to `mesh_z_up.obj`, if you want normal to be correctly rendered in OmniGibson. This can be done with the following commands:
```
meshlabserver -i mesh_z_up.obj -o mesh_z_up_with_normal.obj -om vn vc wt 
mv mesh_z_up_with_normal.obj mesh_z_up.obj
```
4. Create traversable maps, this can be done by running the following commands:

```
python generate_floor_map.py <matterport folder>
python generate_traversable_map.py <matterport folder>

```
5. Move `<matterport folder>` to OmniGibson `dataset`, and you should be able to use it in OmniGibson.
