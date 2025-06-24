# Collision Issues
This guide describes how to address collision/handle complaints.

## Understanding the Requirements
Every object in the BEHAVIOR-1K dataset must be assigned a collision mesh. 

* The collision mesh is the union of a set of convex volumes
* A collision mesh can consist of at most 40 convex volumes
* Each convex volume must have at most 60 vertices
* The different convex volumes comprising a collision mesh are allowed to intersect
one another. This does not cause any issues and can be used to simplify meshes
for gridlike objects.
* The collision mesh must completely represent the object's affordances with the smallest
possible number of convex volumes. For example:
    * Cavities of fillable objects (e.g. bottles, etc.) must NOT be closed off
    * Supporting surfaces e.g. of shelves, cabinets, etc. must be correctly represented: they
    should be flat, correctly sloped
    * Handles should be clearly represented, and if they have a gap underneath e.g. where a finger would normally fit, such gaps should NOT be closed off

**If an object is assigned a collision complaint, its current collision mesh was deemed by a reviewer to have issues with one or more of the above requirements.** It is very important to correctly identify the issue prior to starting work on the object.

## Understanding the Tooling
The asset pipeline contains some tools to allow quick processing of collision meshes. These can be installed by running the `asset_pipeline/b1k_pipeline/add_buttons.py` script from the Scripting > Run Script menu dialog. After running this script a SVL menu will be placed in the main menu of 3ds Max with buttons for each of the below described scripts and more.

### View Complaints
The View Complaints tool shows you unresolved complaints that are attached to the currently selected object, or if no object is selected, it shows you the complaints on all the objects in the current file.

### Resolve Complaints
The Resolve Complaints tool marks as resolved all of the complaints shown in View Complaints for the selected object. It should be used after all work on the object is completed and confirmed.

### Merge Collision
The Merge Collision tool allows you to quickly attach, triangulate, and rename a set of meshes to be assigned as the collision mesh of an object.

To use it, you need to select the collision objects you'd like to merge, then click on the `Merge Collision` button, then select the visual object you want to assign the combined collision mesh to.

### Convex Decomposition
The Convex Decomposition tool runs some off-the-shelf convex decomposition algorithms (mainly CoACD and V-HACD) to generate collision mesh candidates for a selected object. **This is a slow tool (it can take ~30sec for simple objects and up to 10-15 minutes for complex objects).** We therefore recommend considering `Convex Hull` w/ manual subobject selection first.

To run this script, you can just select a visual object, and candidates will be generated for decomposing the object into 4, 8, 16, and 32 volumes for both of CoACD and V-HACD as well as the convex hull. You can then hide all of these meshes, and unhide them one by one to pick which one is the best. We recommend choosing the smallest number that correctly approximates affordances, cavities, and supporting surfaces. After you pick the best option, delete the rest.

If you run the script with certain faces of the visual object selected (e.g. in face or element selection mode of the modify dialog on the right side menu), it will decompose only the selected part. This can be used to easily separate handles etc. when decomposition tools can't easily do this themselves. You can then merge those meshes together using `Merge Collision`.

### Convex Hull
The Convex Hull tool computes the convex hull of the selected mesh, or if a subset of the faces are selected, those faces, and records the outcome as an object underneath the selected visual mesh in the Scene Explorer tree. It is very fast (~1 sec).

If you run the script with certain faces of the visual object selected (e.g. in face or element selection mode of the modify dialog on the right side menu), it will generate the convex hull of only the selected part. This can be used to easily separate handles etc. when decomposition tools can't easily do this themselves. You can then merge those meshes together using `Merge Collision`.

### Validate Collision
The validate collision tool checks that the selected collision mesh satisfies all of the requirements: naming, element count, and vertex count. It's a simplified, fast version of the collision mesh checks inside the file-level sanitycheck. To use it, simply select the collision object and click on the `Validate Collision` button.

## Steps to Follow

Quick note: if any of the steps below fail due to missing dependencies, you can install them by opening the `Scripting > Maxscript Listener` dialog, clicking `Python`, and running e.g. `import pip; pip.main(["install", "coacd"])` where `coacd` is the package you want to install. To run the line you need to press Shift + Enter.

After reading all of the above, follow the below setup steps:

1. `git pull` the BEHAVIOR-1K repository's `main` branch
2. Create your own branch for what you're working on, e.g. `git checkout -b collision`
2. `dvc pull` in the BEHAVIOR-1K repository's asset_pipeline directory to make sure you have the latest files.
3. **Make sure you did BOTH of the above! Otherwise you will be editing an outdated copy of the file and you'll have to redo the work because these files cannot be diffed and merged.**
4. "Unprotect" all of the files. This allows you to edit the files, replacing the symlinks with editable copies of the file. You can run this Powershell command to do that: `dvc unprotect (Get-Item cad/objects/*/processed.max).FullName`

After doing the above, follow the steps described below for each file matching cad/objects/*/processed.max (e.g. in batch-00)

1. Press `View Complaints` to show all complained objects in the file.
2. Search for, and select, the first object. Note that the complaint will be at object level, howwever, if the object complaints multiple links, the problem might be on any one of the links, or all of them.
    * To search for an object, type/paste its 6 character model ID into the search box in the Scene Explorer, and put an asterisk (`*`) before the ID.
    * If you type the asterisk first, the application will likely freeze as you type/paste the rest - not recommended!
3. Carefully evaluate, and understand, what the problem might be that a complaint was added. Review the `Understanding the Requirements` section if needed.
4. Correct the issue. You can do one of the below:
    * Create a collision mesh from scratch. To do this:
        1. Remove the existing collision mesh by deleting the Mcollision object altogether.
        2. Decide if the object can be represented by just taking its convex hull. If so, just hit the `Convex Hull` button to obtain the convex hull as the collision mesh.
        3. If it cannot be represented via a single convex hull, consider if it can be represented by separately computing the convex hulls of certain subsets of its faces/elements. This is usually true for regularly shaped objects like cabinets etc. where the collision mesh can easily be generated by taking the convex hulls of each of the sides of the object. If this is true, select the first such subset of the faces, and click the `Convex Hull` button. Then move onto the next subset, repeat until all subsets have a corresponding collision shape. Then finish by using the `Merge Collision` script to combine the shapes.
        4. If it cannot be generated this way (usually true for irregular shapes e.g. bottles) use the Convex Decomposition tool. Pick one of the candidates it generates and discard the rest.
        5. If none of those candidates is good (e.g. misses affordance, has bad supporting surface, etc.), consider applying the decomposition to subsets of the object's faces, just like explained for Convex Hull above, and merging them at the end.  
    * Make edits to the existing collision mesh
        1. Open the existing collision mesh and remove the parts that are not approximated well. This may not be easy for decomposed objects since the decomposition boundaries are often weird - in that case we recommend starting from scratch.
        2. Follow the steps from the above bullet point to generate collision meshes for the parts that are not represented well.
        3. Use the `Merge Collision` tool to merge the newly generated parts with the remaining parts of the original collision mesh.
5. Make sure you have merged all parts of the collision mesh of each link into a single object, suffixed Mcollision and placed underneath the visual object in the Scene Explorer tree. Use the `Merge Collision` tool to do this. How to use it is documented above. **The script also takes care of renaming the object, as such, you can use it to quickly rename even if you just have one collision mesh.**
6. Select the collision mesh and use the `Validate Collision` tool to validate the collision mesh. Confirm in the MaxScript Listener dialog that the output is VALID. **Then hide/unhide/hide the collision mesh and visually confirm that it correctly represents the object and matches all of the requirements discussed at the top of this document.**
7. If **any link of** the object has a fillable volume also annotated (e.g. a Mfillable object underneath the link) - **you MUST remove it and recreate it!** Delete the Mfillable mesh and follow the steps in the `fillable.md` file in this directory for a description of how to create one.
8. Mark the complaint as resolved by selecting the object and running the `Resolve Complaint` tool.
9. Save the file to make sure you don't lose work.
10. Repeat from Step 1 until there are no complaints left in that file.
11. Save the file once more.
12. Run `dvc add [filename]` where filename is the name of the file you just edited, e.g. `dvc add cad/objects/batch-00/processed.max`. This will make DVC cache the file and update the pointer file to point at your edited version.
13. `git commit` the edited file and also the complaints.json file in the same directory.

To upload your work (e.g. when done, or if you want us to check progress):

1. Run `dvc push` and make sure it finishes
2. Run `git push` to your branch on the BEHAVIOR-1K repository. **Make sure you did step 1 before this, otherwise you'll be pushing just pointers and not actual files!**
3. Open a PR on the BEHAVIOR-1K repository and tag @cgokmen for review.