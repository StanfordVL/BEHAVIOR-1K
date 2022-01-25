

from igibson.objects.stateful_object import StatefulObject


class SoftObject(StatefulObject):
    """
    Soft object (WIP)
    """

    def __init__(
        self,
        filename,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
        scale=-1,
        mass=-1,
        collisionMargin=-1,
        useMassSpring=0,
        useBendingSprings=0,
        useNeoHookean=0,
        springElasticStiffness=1,
        springDampingStiffness=0.1,
        springBendingStiffness=0.1,
        NeoHookeanMu=1,
        NeoHookeanLambda=1,
        NeoHookeanDamping=0.1,
        frictionCoeff=0,
        useFaceContact=0,
        useSelfCollision=0,
        **kwargs
    ):
        super(SoftObject, self).__init__(**kwargs)
        self.filename = filename
        self.scale = scale
        self.basePosition = basePosition
        self.baseOrientation = baseOrientation
        self.mass = mass
        self.collisionMargin = collisionMargin
        self.useMassSpring = useMassSpring
        self.useBendingSprings = useBendingSprings
        self.useNeoHookean = useNeoHookean
        self.springElasticStiffness = springElasticStiffness
        self.springDampingStiffness = springDampingStiffness
        self.springBendingStiffness = springBendingStiffness
        self.NeoHookeanMu = NeoHookeanMu
        self.NeoHookeanLambda = NeoHookeanLambda
        self.NeoHookeanDamping = NeoHookeanDamping
        self.frictionCoeff = frictionCoeff
        self.useFaceContact = useFaceContact
        self.useSelfCollision = useSelfCollision

    def _load(self, simulator):
        """
        Load the object into pybullet
        """
        body_id = p.loadSoftBody(
            self.filename,
            scale=self.scale,
            basePosition=self.basePosition,
            baseOrientation=self.baseOrientation,
            mass=self.mass,
            collisionMargin=self.collisionMargin,
            useMassSpring=self.useMassSpring,
            useBendingSprings=self.useBendingSprings,
            useNeoHookean=self.useNeoHookean,
            springElasticStiffness=self.springElasticStiffness,
            springDampingStiffness=self.springDampingStiffness,
            springBendingStiffness=self.springBendingStiffness,
            NeoHookeanMu=self.NeoHookeanMu,
            NeoHookeanLambda=self.NeoHookeanLambda,
            NeoHookeanDamping=self.NeoHookeanDamping,
            frictionCoeff=self.frictionCoeff,
            useFaceContact=self.useFaceContact,
            useSelfCollision=self.useSelfCollision,
        )

        # Set signed distance function voxel size (integrate to Simulator class?)
        p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.1)

        simulator.load_object_in_renderer(self, body_id, self.class_id, **self._rendering_params, softbody=True)

        return [body_id]

    def add_anchor(self, nodeIndex=-1, bodyUniqueId=-1, linkIndex=-1, bodyFramePosition=[0, 0, 0], physicsClientId=0):
        """
        Create soft body anchor
        """
        p.createSoftBodyAnchor(
            self.get_body_ids()[0], nodeIndex, bodyUniqueId, linkIndex, bodyFramePosition, physicsClientId
        )
