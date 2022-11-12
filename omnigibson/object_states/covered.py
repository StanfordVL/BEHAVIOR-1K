from omnigibson.macros import create_module_macros
from omnigibson.object_states import AABB
from omnigibson.object_states.object_state_base import RelativeObjectState, BooleanState
from omnigibson.systems.system_base import get_element_name_from_system, get_system_from_element_name
from omnigibson.systems.macro_particle_system import VisualParticleSystem, get_visual_particle_systems
from omnigibson.systems.micro_particle_system import FluidSystem, get_fluid_systems
from omnigibson.utils.sampling_utils import raytest_batch
from collections import OrderedDict
import numpy as np

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Value in [0, 1] determining the minimum proportion of particles needed to be present in order for Covered --> True
m.VISUAL_PARTICLE_THRESHOLD = 0.75
m.FLUID_THRESHOLD = 0.1


def check_points_z_proximity_to_object_surface(obj, particle_positions, max_distance=0.01):
    """
    Checks which points from @particle_positions are within @max_distance to @obj's surface in the z direction

    Args:
        obj (EntityPrim): Object whose surface will be used to check points' proximity
        particle_positions ((N, 3) array): positions to check for whether it is close to the object's surface
        max_distance (float): Maximum allowable distance for a point to be considered close to the object surface, in m

    Returns:
        (N,) array: boolean numpy array specifying whether each point lies in the mesh
    """
    # For every point, we raycast downwards to check how far away the distance is
    # Grab the lower z bound for the object and set this as the end points for our raytest we'll apply for all particles
    end_positions = np.array(particle_positions)
    end_positions[:, 2] = obj.aabb[0][2]
    # A point is considered close if:
    # 1. A hit has occurred
    # 2. The body hit belongs to obj
    # 3. The distance of the hit is less than the max_distance
    obj_prim_paths = set([link.prim_path for link in obj.links.values()])
    return np.array([res["hit"] and res["rigidBody"] in obj_prim_paths and res["distance"] < max_distance for res in raytest_batch(
        start_points=particle_positions,
        end_points=end_positions,
        hit_number=0,
    )])


def sample_gridwise_downward_rays_onto_object(obj, ray_spacing=0.01, z_offset=0.001):
    """
    Samples a uniformly spaced grid of rays cast onto the object @obj in the z-axis, returning any valid 0th hits on
    the object

    Args:
        obj (EntityPrim): Object on which to cast rays
        ray_spacing (float): Spacing between sampled rays, in meters
        z_offset (float): Distance from top of object's AABB z boundary from which to cast rays

    Returns:
        tuple:
            - int: number of rays sampled
            - list of dict: sampled raytest results only including those that hit the object
    """
    # Generate start points to use -- this is the gridwise-sampled top surface of the object's AABB plus
    # some small tolerance
    lower, upper = obj.aabb
    x = np.linspace(lower[0], upper[0], int((upper[0] - lower[0]) / ray_spacing))
    y = np.linspace(lower[1], upper[1], int((upper[1] - lower[1]) / ray_spacing))
    n_rays = len(x) * len(y)
    start_points = np.stack([
        np.tile(x, len(y)),
        np.repeat(y, len(x)),
        np.ones(n_rays) * upper[2] + z_offset,
    ]).T
    # Bottom points are the same locations but with the lower z value
    end_points = np.array(start_points)
    end_points[:, 2] = lower[2]
    # Run raytest -- estimated area is the proportion of the overall aabb xy extent area which was hit by the object
    obj_prim_paths = set([link.prim_path for link in obj.links.values()])
    return n_rays, [res for res in raytest_batch(
        start_points=start_points,
        end_points=end_points,
        hit_number=0,
    ) if res["hit"] and res["rigidBody"] in obj_prim_paths]


def get_projected_surface_area_to_z_plane(obj, precision=0.01, z_offset=0.001):
    """
    Checks the projected surface area of the object onto the z-plane

    NOTE: This is a stochastic method; Monte Carlo sampling is conducted: a gridwise raytest batch is executed top-down,
    and the resulting proportion of hits determines the proportion of the xy bounding box area that is assumed to be
    filled with the object.

    Args:
        obj (EntityPrim): Object whose surface should be projected into z-plane
        precision (float): Relative precision to use. This will implicitly determine the number of rays generated
        z_offset (float): Distance from top of object's AABB z boundary from which to cast rays

    Returns:
        float: Estimated projected surface area in the z-plane
    """
    # Determine how to space the rays in order to achieve the desired precision
    # We need at least 1 / precision total rays in order to achieve the precision
    # So we take the minimum between the x and y AABB lengths, and then multiply by the square root of the precision to
    # get the ray spacing to achieve the precision
    ray_spacing = obj.aabb_extent[:2].min() * np.sqrt(precision)
    # Get sampled points that hit the object
    n_points, hits = sample_gridwise_downward_rays_onto_object(obj=obj, ray_spacing=ray_spacing, z_offset=z_offset)

    # Estimated area is the proportion of the overall aabb xy extent area which was hit by the object
    return np.product(obj.aabb_extent[:2]) * len(hits) / n_points


def sample_particle_positions_on_object_top_surface(obj, particle_spacing, z_offset=0.001):
    """
    Samples particle positions on the object @obj's top surface (top wrt the world frame of reference) such that
    its surface would be covered by particles.


    Checks the projected surface area of the object onto the z-plane

    NOTE: This is a stochastic method; Monte Carlo sampling is conducted: a gridwise raytest batch is executed top-down,
    and the resulting proportion of hits determines the proportion of the xy bounding box area that is assumed to be
    filled with the object.

    Args:
        obj (EntityPrim): Object whose surface should be projected into z-plane
        particle_spacing (float): Distance between sampled particle positions
        z_offset (float): Distance in the z-axis to offset the particles from the object surface, in addition to the
            desired @particle_spacing distance

    Returns:
        (N, 3) array: Sampled particle positions for covering the object's top surface
    """
    # Sample rays to determine where a valid particle can be sampled
    _, hits = sample_gridwise_downward_rays_onto_object(obj=obj, ray_spacing=particle_spacing, z_offset=particle_spacing)

    # Convert the hits into a numpy array of values
    particle_positions = np.array([hit["position"] for hit in hits])
    particle_positions[:, 2] = particle_positions[:, 2] + particle_spacing + z_offset

    return particle_positions


class Covered(RelativeObjectState, BooleanState):
    def __init__(self, obj):
        # Run super first
        super().__init__(obj)

        # Set internal values
        self._visual_particle_groups = None
        self._n_initial_visual_particles = None

    @staticmethod
    def get_dependencies():
        # AABB needed for sampling visual particles on an object
        return RelativeObjectState.get_dependencies() + [AABB]

    @property
    def stateful(self):
        return True

    def _initialize(self):
        # Create the visual particle groups
        self._visual_particle_groups = OrderedDict((get_element_name_from_system(system), system.create_attachment_group(obj=self.obj))
                                                   for system in get_visual_particle_systems().values())

        # Default initial particles is 0
        self._n_initial_visual_particles = OrderedDict((get_element_name_from_system(system), 0)
                                                       for system in get_visual_particle_systems().values())

    def _get_value(self, system):
        # Value is false by default
        value = False
        # First, we check what type of system
        # Currently, we support VisualParticleSystems and FluidSystems
        if issubclass(system, VisualParticleSystem):
            # We check whether the current number of particles assigned to the group is greater than the threshold
            name = get_element_name_from_system(system)
            value = system.num_group_particles(group=self._visual_particle_groups[name]) \
                   > m.VISUAL_PARTICLE_THRESHOLD * self._n_initial_visual_particles[name]
        elif issubclass(system, FluidSystem):
            # We only check if we have particle instancers currently
            if len(system.particle_instancers) > 0:
                # We've already cached particle contacts, so we merely search through them to see if any particles are
                # touching the object
                particle_width = system.particle_radius * 2
                n_near_particles = np.sum([len(idxs) for idxs in system.state_cache["particle_contacts"][self.obj].values()])
                # Heuristic: Assuming each particle has net surface area coverage of particle_width ^ 2 (i.e.: square),
                # We find the total area coverage proportion with respect to the bird's eye view area of the object
                area_covered = n_near_particles * (particle_width ** 2)
                total_area = get_projected_surface_area_to_z_plane(obj=self.obj, precision=0.01)
                value = area_covered / total_area > m.FLUID_THRESHOLD
        else:
            raise ValueError(f"Invalid system {system} received for getting Covered state!"
                             f"Currently, only VisualParticleSystems and FluidSystems are supported.")

        return value

    def _set_value(self, system, new_value):
        # Default success value is True
        success = True
        # First, we check what type of system
        # Currently, we support VisualParticleSystems and FluidSystems
        if issubclass(system, VisualParticleSystem):
            # Check current state and only do something if we're changing state
            if self.get_value(system) != new_value:
                name = get_element_name_from_system(system)
                group = self._visual_particle_groups[name]
                if new_value:
                    # Generate particles
                    success = system.generate_group_particles(group=group)
                    # If we succeeded with generating particles (new_value = True), store additional info
                    if success:
                        # Store how many particles there are now -- this is the "maximum" number possible
                        self._n_initial_visual_particles[name] = system.num_group_particles(group=group)
                else:
                    # We remove all of this group's particles
                    system.remove_all_group_particles(group=group)

        elif issubclass(system, FluidSystem):
            # Check current state and only do something if we're changing state
            if self.get_value(system) != new_value:
                if new_value:
                    # We first shoot grid-wise rays downwards onto the object
                    # For any rays that hit the object, we determine the appropriate z distance that is slightly offset
                    # above the object's surface and store those positions
                    # Then we sample particles at each of those positions
                    particle_positions = sample_particle_positions_on_object_top_surface(
                        obj=self.obj,
                        particle_spacing=system.particle_radius * 2,
                        z_offset=0.001,
                    )
                    system.generate_particle_instancer(
                        n_particles=len(particle_positions),
                        positions=particle_positions,
                    )
                else:
                    # We hide all particles within range to be garbage collected by fluid system
                    for inst in system.particle_instancers.values():
                        inst.particle_visibilities = 1 - check_points_z_proximity_to_object_surface(
                            obj=self.obj,
                            particle_positions=inst.particle_positions,
                            max_distance=system.particle_radius * 2,
                        )

        else:
            raise ValueError(f"Invalid system {system} received for setting Covered state!"
                             f"Currently, only VisualParticleSystems and FluidSystems are supported.")

        return success

    @property
    def state_size(self):
        # We have a single value for every visual particle system
        return len(get_visual_particle_systems())

    @property
    def _supported_systems(self):
        """
        Returns:
            list: All systems used in this state, ordered deterministically
        """
        return list(get_visual_particle_systems().values()) + list(get_fluid_systems().values())

    def _dump_state(self):
        # For every visual particle system, add the initial number of particles
        state = OrderedDict()
        for system in get_visual_particle_systems().values():
            name = get_element_name_from_system(system)
            state[f"{name}_initial_visual_particles"] = self._n_initial_visual_particles[name]

        return state

    def _load_state(self, state):
        # For every visual particle system, set the initial number of particles
        for system in get_visual_particle_systems().values():
            name = get_element_name_from_system(system)
            self._n_initial_visual_particles[name] = state[f"{name}_initial_visual_particles"]

    def _serialize(self, state):
        return np.array([val for val in state.values()], dtype=float)

    def _deserialize(self, state):
        state_dict = OrderedDict()
        for i, system in enumerate(get_visual_particle_systems().values()):
            name = get_element_name_from_system(system)
            state_dict[f"{name}_initial_visual_particles"] = int(state[i])

        return state_dict, len(state_dict)
