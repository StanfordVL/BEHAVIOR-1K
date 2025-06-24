import time
from pathlib import Path

import mujoco
import mujoco.viewer


def main():
    _PROJECT_ROOT: Path = Path(__file__).parent.parent
    _MENAGERIE_ROOT: Path = _PROJECT_ROOT / "third_party" / "mujoco_menagerie"
    xml = _MENAGERIE_ROOT / "franka_emika_panda" / "panda.xml"

    # xml = _MENAGERIE_ROOT / "universal_robots_ur5e" / "ur5e.xml"

    m = mujoco.MjModel.from_xml_path(xml.as_posix())
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        while viewer.is_running():
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            # Example modification of a viewer option: toggle contact points every two seconds.
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
