from omnigibson.physics_backends.newton_backend import NewtonBackend
from omnigibson.physics_backends.physx_backend import PhysXBackend

PHYSICS_BACKENDS = {
    "physx": PhysXBackend,
    "newton": NewtonBackend,
}
