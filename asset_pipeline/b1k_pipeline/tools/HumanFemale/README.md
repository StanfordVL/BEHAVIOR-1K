This asset demonstrates use of UsdSkel schemas for describing a
skeletally-deformed character in USD.

The assets are broken down into the following components:

# HumanFemale.usd

This file is the core 'asset', defining both the geometry that makes up the
HumanFemale character,its materials as well as all of the UsdSkelBindingAPI
properties that control the way that a Skeleton affects geometry.

The example also defines several variant sets on scope </HumanFemale_Group>,
demonstrating possible applications of variant sets:

## geometricLOD:
    
A variant set for swtiching between three different levels of geometric
complexity. The skel bindings have been authored for each level of geometric
complexity, so that linear blend skinning still applies to the character when
selecting a different LOD level.

## rigComplexity:

A variant set used to switch between three differents levels of 'rig' complexity.
The 'reduced' complexity rig excludes minor details like fingers, blendshapes, or 
face joints, while the default 'high' complexity includes a complete set of joints
with facial blendshapes.  The 'faceBones' variant uses joints instead of blendshapes
for facial deformation.

# Animation Clips: HumanFemale.keepAlive.usd, HumanFemale.walk.usd

Both of these layer provide simple animation clips.
These animation clips contain only joint transforms, and root character
transform animation.

For the sake of visualizing the animation, animation clips reference in the
common 'HumanFemale.usd' asset, so that there is skinnable geometry available
for preview. This is not strictly necessary when defining animation, however:
It is valid to author animation clips that contain only joint animations.

# Payloads:

In addition to providing a complete example of use of UsdSkel schemas, this
asset has been setup to make use of payloads. The HumanFemale assembly has been
broken into a set of sub-models ('components'), each of which is referenced
in through a payload.

There are many different ways of organizing assets, but for the purposes of this
example, each payload has been placed in a layer corresponding to each
different geometryLOD variant.

For each geometricLOD ('low', 'medium', 'full'),
`assets/HumanFemale.{LOD}_payload.usd` includes the payload contents.
I.e, the geometry, shaders, skel binding properties, etc.
Paired with these payload files, each `assets/HumanFemale.{LOD}.usd` file
applies the payload arcs for the corresponding LOD variant. In Pixar's pipeline,
each payload arc would normally target a unique file for each of the character, 
hair, and garment assets, but for pedantic simplicity we have bundled all data 
into a single file, and made each payload arc target a sub-root prim of that
file, which means that all data on the root </HumanFemale_Group> prim in the
`HumanFemale.{LOD}_payload.usd` files is ignored.

# assets/HumanFemale.rig.usd

The rig.usd file defines HumanFemale's Skeleton, along with its bind pose
and rest pose. Multiple configurations of the Skeleton have been defined
at this point within the 'rigComplexity' variant set.

In production pipelines, some form of the rig.usd file might contain a common
definition of a Skeleton -- and possibly other site-specific rig components --
which could then be reused across multiple character types using referencing
and other composition features.

# Blendshapes:

For each geometricLOD ('low', 'medium', 'full'),
`assets/HumanFemale.{LOD}.blendshapes.usd` contains the blendshape data used
by the default rig variant (high).  The full blendshapes usd contains a very
large set of shapes designed for high detail.  The medium blendshapes usd
sublayers the full blendshapes and updates the attributes to apply to the medium
LOD topology.  The low blendshapes usd does the same, but for the low LOD topology.
In addition, as a demonstration of how to non-destructively reduce the number of
blendshapes for rig LOD, the assets/HumanFemale.low_payload.usd deactivates the
majority of blendshapes and updates the top level blendshape attributes accordingly.
Blendshape weight animation is contained in the animation clips.



