"""UR5e composer class."""

from pathlib import Path
from typing import Optional, Union

from dm_control import mjcf

from gello.dm_control_tasks.arms.manipulator import Manipulator
from gello.dm_control_tasks.mjcf_utils import MENAGERIE_ROOT


class UR5e(Manipulator):
    GRIPPER_XML = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"
    XML = MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"

    def _build(
        self,
        name: str = "UR5e",
        xml_path: Union[str, Path] = XML,
        gripper_xml_path: Optional[Union[str, Path]] = GRIPPER_XML,
    ) -> None:
        super()._build(name=name, xml_path=xml_path, gripper_xml_path=gripper_xml_path)

    @property
    def flange(self) -> mjcf.Element:
        return self._mjcf_root.find("site", "attachment_site")
