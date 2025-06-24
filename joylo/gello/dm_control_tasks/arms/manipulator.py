"""Manipulator composer class."""

import abc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.mujoco.wrapper import mjbindings
from dm_control.suite.utils.randomizers import random_limited_quaternion

from gello.dm_control_tasks import mjcf_utils


def attach_hand_to_arm(
    arm_mjcf: mjcf.RootElement,
    hand_mjcf: mjcf.RootElement,
    # attach_site: str,
) -> None:
    """Attaches a hand to an arm.

    The arm must have a site named "attachment_site".

    Taken from https://github.com/deepmind/mujoco_menagerie/blob/main/FAQ.md#how-do-i-attach-a-hand-to-an-arm

    Args:
      arm_mjcf: The mjcf.RootElement of the arm.
      hand_mjcf: The mjcf.RootElement of the hand.
      attach_site: The name of the site to attach the hand to.

    Raises:
      ValueError: If the arm does not have a site named "attachment_site".
    """
    physics = mjcf.Physics.from_mjcf_model(hand_mjcf)

    # attachment_site = arm_mjcf.find("site", attach_site)
    attachment_site = arm_mjcf.find("site", "attachment_site")
    if attachment_site is None:
        raise ValueError("No attachment site found in the arm model.")

    # Expand the ctrl and qpos keyframes to account for the new hand DoFs.
    arm_key = arm_mjcf.find("key", "home")
    if arm_key is not None:
        hand_key = hand_mjcf.find("key", "home")
        if hand_key is None:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, np.zeros(physics.model.nu)])
            arm_key.qpos = np.concatenate([arm_key.qpos, np.zeros(physics.model.nq)])
        else:
            arm_key.ctrl = np.concatenate([arm_key.ctrl, hand_key.ctrl])
            arm_key.qpos = np.concatenate([arm_key.qpos, hand_key.qpos])

    attachment_site.attach(hand_mjcf)


class Manipulator(composer.Entity, abc.ABC):
    """A manipulator entity."""

    def _build(
        self,
        name: str,
        xml_path: Union[str, Path],
        gripper_xml_path: Optional[Union[str, Path]],
    ) -> None:
        """Builds the manipulator.

        Subclasses can not override this method, but should call this method in their
        own _build() method.
        """
        self._mjcf_root = mjcf.from_path(str(xml_path))
        self._arm_joints = tuple(mjcf_utils.safe_find_all(self._mjcf_root, "joint"))
        if gripper_xml_path:
            gripper_mjcf = mjcf.from_path(str(gripper_xml_path))
            attach_hand_to_arm(self._mjcf_root, gripper_mjcf)

        self._mjcf_root.model = name
        self._add_mjcf_elements()

    def set_joints(self, physics: mjcf.Physics, joints: np.ndarray) -> None:
        assert len(joints) == len(self._arm_joints)
        for joint, joint_value in zip(self._arm_joints, joints):
            joint_id = physics.bind(joint).element_id
            joint_name = physics.model.id2name(joint_id, "joint")
            physics.named.data.qpos[joint_name] = joint_value

    def randomize_joints(
        self,
        physics: mjcf.Physics,
        random: Optional[np.random.RandomState] = None,
    ) -> None:
        random = random or np.random  # type: ignore
        assert random is not None
        hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
        slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE
        ball = mjbindings.enums.mjtJoint.mjJNT_BALL
        free = mjbindings.enums.mjtJoint.mjJNT_FREE

        qpos = physics.named.data.qpos

        for joint in self._arm_joints:
            joint_id = physics.bind(joint).element_id
            # joint_id = physics.model.name2id(joint.name, "joint")
            joint_name = physics.model.id2name(joint_id, "joint")
            joint_type = physics.model.jnt_type[joint_id]
            is_limited = physics.model.jnt_limited[joint_id]
            range_min, range_max = physics.model.jnt_range[joint_id]

            if is_limited:
                if joint_type in [hinge, slide]:
                    qpos[joint_name] = random.uniform(range_min, range_max)

                elif joint_type == ball:
                    qpos[joint_name] = random_limited_quaternion(random, range_max)

            else:
                if joint_type == hinge:
                    qpos[joint_name] = random.uniform(-np.pi, np.pi)

                elif joint_type == ball:
                    quat = random.randn(4)
                    quat /= np.linalg.norm(quat)
                    qpos[joint_name] = quat

                elif joint_type == free:
                    # this should be random.randn, but changing it now could significantly
                    # affect benchmark results.
                    quat = random.rand(4)
                    quat /= np.linalg.norm(quat)
                    qpos[joint_name][3:] = quat

    def _add_mjcf_elements(self) -> None:
        # Parse joints.
        joints = mjcf_utils.safe_find_all(self._mjcf_root, "joint")
        joints = [joint for joint in joints if joint.tag != "freejoint"]
        self._joints = tuple(joints)

        # Parse actuators.
        actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")
        self._actuators = tuple(actuators)

        # Parse qpos / ctrl keyframes.
        self._keyframes = {}
        keyframes = mjcf_utils.safe_find_all(self._mjcf_root, "key")
        if keyframes:
            for frame in keyframes:
                if frame.qpos is not None:
                    qpos = np.array(frame.qpos)
                    self._keyframes[frame.name] = qpos

        # add a visualizeation the flange position that is green
        self.flange.parent.add(
            "geom",
            name="flange_geom",
            type="sphere",
            size="0.01",
            rgba="0 1 0 1",
            pos=self.flange.pos,
            contype="0",
            conaffinity="0",
        )

    def _build_observables(self):
        return ArmObservables(self)

    @property
    @abc.abstractmethod
    def flange(self) -> mjcf.Element:
        """Returns the flange element.

        The flange is the end effector of the manipulator where tools can be
        attached, such as a gripper.
        """

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @property
    def joints(self) -> Tuple[mjcf.Element, ...]:
        return self._joints

    @property
    def actuators(self) -> Tuple[mjcf.Element, ...]:
        return self._actuators

    @property
    def keyframes(self) -> Dict[str, np.ndarray]:
        return self._keyframes


class ArmObservables(composer.Observables):
    """Base class for quadruped observables."""

    @composer.observable
    def joints_pos(self):
        return observable.MJCFFeature("qpos", self._entity.joints)

    @composer.observable
    def joints_vel(self):
        return observable.MJCFFeature("qvel", self._entity.joints)

    @composer.observable
    def flange_position(self):
        return observable.MJCFFeature("xpos", self._entity.flange)

    @composer.observable
    def flange_orientation(self):
        return observable.MJCFFeature("xmat", self._entity.flange)

    # Semantic grouping of observables.
    def _collect_from_attachments(self, attribute_name: str):
        out: List[observable.MJCFFeature] = []
        for entity in self._entity.iter_entities(exclude_self=True):
            out.extend(getattr(entity.observables, attribute_name, []))
        return out

    @property
    def proprioception(self):
        return [
            self.joints_pos,
            self.joints_vel,
            self.flange_position,
            # self.flange_orientation,
            # self.flange_velocity,
            # self.flange_angular_velocity,
        ] + self._collect_from_attachments("proprioception")
