---
icon: material/car-wrench
---

# Under the Hood: Isaac Sim Details
In this page, we discuss the particulars of certain Isaac Sim features and behaviors.

## Playing and Stopping
TODO

## CPU and GPU dynamics and pipelines
TODO

## Sources of Truth: USD, PhysX and Fabric
In Isaac Sim, there are three competing representations of the current state of the scene: USD, PhysX, and Fabric. These are used in different contexts, with USD being the main source of truth for loading and representing the scene, PhysX only being used opaquely during physics simulation, and Fabric providing a faster source of truth for the renderer during physics simulation.

### USD
USD is the scene graph representation of the scene, as directly loaded from the USD files. This is the main scene / stage representation used by Omniverse apps.

  * This representation involves maintaining the full USD tree in memory and mutating it as the scene changes.
  * It is a complete, flexible representation containing all scene meshes and hierarchy that works really well for representing static scenes (e.g. no realtime physics simulation), e.g. for usual CAD workflows.
  * During physics simulation, we need to repeatedly update the transforms of the objects in the scene so that they will be rendered in their new poses. USD is not optimized for this: especially due to specific USD features like transforms being defined locally (so to compute a world transform, you need to traverse the tree). Queries and reads/writes using the Pixar USD library are also overall relatively slow.

### PhysX
PhysX contains an internal physics-only representation of the scene that it uses to perform computations during physics simulation.

  * The PhysX representation is only available when simulation is playing (e.g. when it is stopped, all PhysX internal storage is freed, and when it is played again, the scene is reloaded from USD).
  * This representation is the fastest source for everything it provides (e.g. transforms, joint states, etc.) since it only contains physics-relevant information and provides methods to access these in a tensorized manner, called tensor APIs, used in a number of places in OmniGibson.
  * But it does not contain any rendering information and is not available when simulation is stopped. As such, it cannot be used as the renderer as the source of truth.
  * Therefore, by default, PhysX explicitly updates the USD state after every step so that the renderer and the representation of the scene in the viewport are updated. This is a really slow operation for large scenes, causing frame rates to drop below 10 fps even for our smallest scenes.

### Fabric
Fabric (formerly Flatcache) is an optimized representation of the scene that is a flattened version of the USD scene graph that is optimized for fast accesses to transforms and for rendering.

  * It can be enabled using the ENABLE_FLATCACHE global macro in OmniGibson, which causes the renderer to use Fabric to get object transforms instead of USD, and causes PhysX to stop updating the USD state after every step and update the Fabric state instead.
  * The Fabric state exists alongside the USD and captures much of the same information, although it is not as complete as USD. It is optimized for fast reads and writes of object transforms and is used by the renderer to render the scene.
  * The information it contains is usually fresher than the USD, e.g. when Fabric is enabled, special attention needs to be paid in order to not accidentally access stale information from USD instead of Fabric.
  * Fabric stores world transforms directly, e.g. any changes of a transform of an object's parent will not be reflected in the child's position because the child separately stores its world transform. One main advantage of this setup is that it is not necessary to traverse the tree to compute world transforms.
  * A new library called `usdrt` provides an interface that can be used to access Fabric state in a way that is similar to the Pixar USD library. This is used in a number of places in OmniGibson to access Fabric state.

To conclude, with ENABLE_FLATCACHE enabled, there will be three concurrent representations of the scene state in OmniGibson. USD will be the source of truth for the meshes and the hierarchy. While physics simulation is playing, PhysX will be the source of truth for the physics state of the scene, and we will use it for fast accesses to compute controls etc., and finally on every render step, PhysX will update Fabric which will then be the source of truth for the renderer and for the OmniGibson pose APIs.

The ENABLE_FLATCACHE macro is recommended to be enabled since large scenes will be unplayable without it, but it can be disabled for small scenes, in which case the Fabric representation will not be used, PhysX will update the USD's local transforms on every step, and the renderer will use USD directly.

## Lazy Imports
Almost all of OmniGibson's simulation functionality uses Isaac Sim code, objects, and components to function. These Python components often need to be imported e.g. via an `import omni.isaac.core.utils.prims` statement. However, such imports of Omniverse libraries can only be performed if the Isaac Sim application has already been launched. Launching the application takes up to 10 minutes on the first try due to shader compilation, and 20 seconds every time after that, and requires the presence of a compatible GPU and permissions. However, certain OmniGibson functionality (e.g. downloading datasets, running linters, etc.) does not require the actual _execution_ of any Isaac Sim code, and should not be blocked by the need to import Isaac Sim libraries.

To solve this problem, OmniGibson uses a lazy import system. The `omnigibson.lazy` module, often imported as `import omnigibson.lazy as lazy` provides an interface that only imports modules when they are first used.

Thus, there are two important requirements enforced in OmniGibson with respect to lazy imports:

1. All imports of omni, pxr, etc. libraries should happen through the `omnigibson.lazy` module. Classes and functions can then be accessed using their fully qualified name. For example, instead of `from omni.isaac.core.utils.prims import get_prim_at_path` and then calling `get_prim_at_path(...)`, you should first import the lazy import module `import omnigibson.lazy as lazy` and then call your function using the full name `lazy.omni.isaac.core.utils.prims.get_prim_at_path(...)`.
2. No module except `omnigibson/utils/deprecated_utils.py` should import any Isaac Sim modules at load time (that module is ignored by docs, linters, etc.). This is to ensure that the OmniGibson package can be imported and used without the need to launch Isaac Sim. Instead, Isaac Sim modules should be imported only when they are needed, and only in the functions that use them. If a class needs to inherit from a class in an Isaac Sim module, the class can be placed in the deprecated_utils.py file, or it can be wrapped in a function to delay the import, like in the case of simulator.py.