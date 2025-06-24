# Fillable Volumes

## Understanding the Requirements

In BEHAVIOR-1K, certain object categories/synsets are annotated as `fillable` objects and these objects need their fillable volume annotated. A fillable object might be obvious ones like bottles etc. that need to be filled with fluids, but it might also be a container furniture object such as a cabinet. To be able to efficiently check if something is inside an object and efficiently sample objects there.

* These are the requirements of a fillable mesh:
    * It should cover as much of the object's cavity as possible, and should be in near contact with the bottom. If it is too far from covering the whole cavity, it will cause issues with filledness detection.
    * **Just like a collision mesh, it should consist of convex volumes (up to 40), and each convex volume should contain less than 60 vertices.**
    * **It should **NOT** intersect/go through the object's collision mesh in any way.** The collision and fillable volumes should be fully disjoint.
    * It should end near the top of the cavity, but it can be slightly above or slightly below.
    * If the object is meant to contain fluids (e.g. a sink), you should ONLY annotate the part that's meant to contain a fluid (e.g. not a cabinet underneath).
    * If there are multiple disjoint cavities in the object, the fillable mesh should similarly contain disjoint pieces - it should NOT include the area that is not part of either cavity!

* One key part of the fillable annotation is **where** the annotation is, that is, whether the annotation will move with the base link (e.g. in a washing machine) or with one of the links (e.g. in a drawer unit). This should be carefully selected.

### Open or Closed?
There are two kinds of for generating fillable meshes: the default, "enclosed" option and the "open" option. The difference is that, in the enclosed option, the rays must actually hit a part of the object, requiring and guaranteeing that the annotated location is somewhat enclosed. In the open option, rays are allowed to hit the object's convex hull too, guaranteeing a mesh, but these may be overapproximations, and since this method will also work with areas that are not properly enclosed, volumes generated this way will be flagged so that we don't use them for sampling fluids etc. - as a result, you should refrain from using the OPEN mode unless absolutely necessary e.g. in the case of shelf units.

## Understanding the Tooling
The main set of tools used for fillable volume annotation are the below:

### Generate Fillable Volume
This script starts from a desired seed position node placed underneath an object in the Scene Explorer tree, and shoots rays around to identify the boundary of the cavity the seed position is placed in. It then obtains the hit points and fits a convex hull to them, and places this object as the fillable mesh of the given link. The seed position can then be moved and this script run again, and the ray casting will be repeated and the new results added to the fillable mesh. This way, the seed position can be moved to different cavities and the script repeatedly executed. Note that rays can hit not only the seed's parent object, but also all of the links of the same object that are children of the selected link in the kinematic tree, e.g. if you place the seed under a base link (e.g. the base of a cabinet), rays will be allowed to hit not only the base link but all moving links (e.g. the door) as well.

### Generate Open Fillable Volume
This script does the same as the above, but it allows rays to hit the convex hull of the object as well as its actual faces. It is useful for generating cavities for open-sided container objects like bookcases. See the "Open or Closed" section for more information.

### Add Fillable Seed
When you run this script after selecting an object, a point object will be generated at the center of the object's AABB for you to move around and use as the fillable seed for Generate Fillable Volume.

### Merge Collision
Originally meant for collision meshes, the Merge Collision script can be used to combine multiple 3ds Max objects into a single object and mark it as the fillable mesh of a selected object. To use it, you need to select the objects you'd like to merge, then click on the `Merge Collision` button, then select the object you want to assign the combined mesh to. Note that by default it will name the mesh `Mcollision` assuming it's a collision mesh - make sure to rename it to `Mfillable` for a fillable mesh.

## Steps to Follow
Pre-steps: see if the object actually has an accessible cavity (e.g. the collision mesh doesnt block it). If it doesn't, we might want to fix the object if it's the only one of its kind (see knowledgebase website), or if not, we can delete the object.

## Option 1: the fillable volume generator
1. Pick the link you are doing this for. You need to pick the link that the fillable volume will move with. For example, a cabinet with doors will have its fillable annotation on the base link. A drawer unit will have a separate one on each of its drawers. Your object might be a combination of these (e.g. both the base link AND the drawers will have volumes)
2. Use the `Add Fillable Seed` button to add the seed
3. For each fillable cavity of the object:
  1. Move the seed to roughly the middle of each cavity (use the 1d axes or 2d planes on the move tool, dont move in 3d directly. makes it hard)
  2. Click the Generate Fillable Volume or Generate Open Fillable Volume button (more on this above)
4. Delete the fillable seed object
5. Select the fillable volume and run the `Validate Collision` script which can also validate fillable volumes.

## Option 2: manual annotation
If the generator won't generate a mesh, if the object has a shape that's suitable for it, you can manually create the mesh using convex primitives like box, cone etc. Make sure you don't over or underapproximate, and most importantly, that you don't include any external volume as part of the fillable volume. 

1. Approximate each of the cavities of the object using primitive shapes like Box, Line, Cylinder, etc. or a hand-drawn mesh if you like.
2. After creating the collision mesh out of primitive shapes, use the `Merge Collision` tool to merge the volumes (see the tool description above for how to use), and make sure you rename the final object to Mfillable
3. Select the fillable volume and run the `Validate Collision` script which can also validate fillable volumes.
