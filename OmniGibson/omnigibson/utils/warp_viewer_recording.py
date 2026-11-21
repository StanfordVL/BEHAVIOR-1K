"""Robot teleoperation/recording loop using the global Warp render backend."""

import time

import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.newton_viewer_recording import _PygletTeleopController


def run_with_warp_viewer(env, robot, output=None, num_steps=100, headless=False, control_mode="teleop"):
    """Control the robot; scene rendering and mouse navigation belong to the backend."""
    headless = headless or gm.HEADLESS or output is not None
    writer = controller = None
    backend = og.sim.render_backend
    try:
        if output:
            import imageio.v2 as imageio

            writer = imageio.get_writer(output, fps=30, macro_block_size=1)
        if not headless:
            from pyglet.window import key

            og.sim.render()
            window = backend._viewer.window
            controller = _PygletTeleopController(robot)
            window.push_handlers(controller.on_key_press, controller.on_key_release)

            def on_key_press(symbol, modifiers):
                if symbol == key.R:
                    env.reset()

            window.push_handlers(on_key_press=on_key_press)
            controller.print_teleop_info()
        step = 0
        action = None
        while not headless or step < num_steps:
            started = time.monotonic()
            if headless or control_mode == "random":
                if step % 30 == 0:
                    action = (th.rand(robot.action_dim, device=env.device) * 2 - 1) * 0.05
            else:
                action = controller.get_teleop_action()
            env.step(action)
            if writer:
                obs, _ = og.sim.viewer_camera.get_obs()
                frame = obs["rgb"]
                frame = frame.cpu().numpy() if isinstance(frame, th.Tensor) else frame
                writer.append_data(frame[..., :3])
            if not headless:
                time.sleep(max(0, 1 / 30 - (time.monotonic() - started)))
            step += 1
        print(f"Warp pt1_oidn: completed {step} steps" + (f", saved {output}" if output else ""), flush=True)
    finally:
        if writer:
            writer.close()
