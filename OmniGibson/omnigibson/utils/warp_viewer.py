"""Mouse navigation and sidebar for the standalone Warp render backend."""

import numpy as np


class _WarpViewer:
    """Mouse-operated camera and a read-only sidebar; keyboard events belong to teleop."""

    def __init__(self, camera, width, height):
        import pyglet

        self.camera = camera
        self.width, self.height = width, height
        self.dragging_view = False
        self.control_hint = "Esc: quit"
        self.window = pyglet.window.Window(width + 280, height, caption="OmniGibson — Warp pt1_oidn", resizable=True)
        self.window.set_minimum_size(640, 540)
        self.window.push_handlers(self.on_mouse_press, self.on_mouse_release, self.on_mouse_drag, self.on_mouse_scroll)
        self.background = pyglet.shapes.Rectangle(width, 0, 280, height, color=(28, 31, 38))
        self.label = pyglet.text.Label(
            "",
            x=width + 16,
            y=height - 16,
            width=248,
            multiline=True,
            anchor_y="top",
            font_size=11,
            color=(225, 230, 240, 255),
        )
        self.window.push_handlers(on_resize=self.on_resize)

    def on_resize(self, width, height):
        self.width, self.height = max(1, width - 280), max(1, height)
        self.background.x = self.width
        self.background.height = self.height
        self.label.x = self.width + 16
        self.label.y = self.height - 16

    def on_mouse_press(self, x, y, button, modifiers):
        self.dragging_view = 0 <= x < self.width and 0 <= y < self.height

    def on_mouse_release(self, x, y, button, modifiers):
        self.dragging_view = False

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        from pyglet.window import mouse

        if not self.dragging_view:
            return
        pos = np.asarray(self.camera["pos"], dtype=float)
        target = np.asarray(self.camera["look_at"], dtype=float)
        offset = pos - target
        distance = np.linalg.norm(offset)
        if buttons & mouse.MIDDLE:
            forward = -offset / distance
            right = np.cross(forward, [0, 0, 1])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            scale = 2 * distance * np.tan(np.radians(self.camera["fov"]) / 2) / self.height
            delta = (-dx * right - dy * up) * scale
            pos, target = pos + delta, target + delta
        elif buttons & (mouse.LEFT | mouse.RIGHT):
            yaw = np.arctan2(offset[1], offset[0]) - dx * 0.005
            pitch = np.clip(np.arcsin(offset[2] / distance) + dy * 0.005, -1.5, 1.5)
            offset = distance * np.array(
                [
                    np.cos(pitch) * np.cos(yaw),
                    np.cos(pitch) * np.sin(yaw),
                    np.sin(pitch),
                ]
            )
            if buttons & mouse.LEFT:
                pos = target + offset  # Orbit the current target.
            else:
                target = pos - offset  # Look around from the current position.
        self.camera.update(pos=pos, look_at=target)

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        target = np.asarray(self.camera["look_at"], dtype=float)
        offset = np.asarray(self.camera["pos"], dtype=float) - target
        distance = np.linalg.norm(offset)
        new_distance = np.clip(distance * np.exp(np.clip(-scroll_y * 0.12, -10, 10)), 0.1, 100)
        self.camera["pos"] = target + offset * (new_distance / distance)

    def draw(self, frame, robot, step, fps, step_ms, render_ms):
        import pyglet

        self.window.clear()
        frame_height, frame_width = frame.shape[:2]
        scale = min(self.width / frame_width, self.height / frame_height)
        display_width, display_height = frame_width * scale, frame_height * scale
        pyglet.image.ImageData(frame_width, frame_height, "RGB", frame.tobytes(), pitch=-frame_width * 3).blit(
            (self.width - display_width) / 2,
            (self.height - display_height) / 2,
            width=display_width,
            height=display_height,
        )
        self.background.draw()
        camera_xyz = ", ".join(f"{v:.2f}" for v in self.camera["pos"])
        robot_xyz = ", ".join(f"{v:.2f}" for v in robot.get_position_orientation()[0]) if robot else "n/a"
        step_timing = f"{step_ms:.1f} ms" if step_ms is not None else "n/a"
        self.label.text = (
            "WARP VIEWER\n"
            "Physics: Newton\nRenderer: pt1_oidn\nOIDN: high | 1 sample/pixel\n"
            f"Render resolution: {frame_width} x {frame_height}\n"
            f"Viewer: {fps:.1f} FPS\nEnvironment step: {step_timing}\n"
            f"Render + denoise: {render_ms:.1f} ms\nStep: {step}\n"
            f"Robot: {robot.name if robot else 'none'}\nXYZ: {robot_xyz}\nCamera: {camera_xyz}\n\n"
            "MOUSE (over scene)\nLeft drag: orbit\nRight drag: look around\n"
            "Middle drag: pan\nScroll: dolly in / out\n\n"
            "KEYBOARD\nReserved for robot teleop\nSee terminal for robot bindings\n"
            f"{self.control_hint}"
        )
        self.label.draw()
        self.window.flip()
