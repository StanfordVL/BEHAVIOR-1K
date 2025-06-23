"""Franka composer class."""

from pathlib import Path
from typing import Optional, Union

from dm_control import mjcf

from gello.dm_control_tasks.arms.manipulator import Manipulator
from gello.dm_control_tasks.mjcf_utils import MENAGERIE_ROOT

XML = MENAGERIE_ROOT / "franka_emika_panda" / "panda_nohand.xml"
GRIPPER_XML = MENAGERIE_ROOT / "robotiq_2f85" / "2f85.xml"


class Franka(Manipulator):
    """Franka Robot."""

    def _build(
        self,
        name: str = "franka",
        xml_path: Union[str, Path] = XML,
        gripper_xml_path: Optional[Union[str, Path]] = GRIPPER_XML,
    ) -> None:
        super()._build(name="franka", xml_path=XML, gripper_xml_path=GRIPPER_XML)

    @property
    def flange(self) -> mjcf.Element:
        return self._mjcf_root.find("site", "attachment_site")
