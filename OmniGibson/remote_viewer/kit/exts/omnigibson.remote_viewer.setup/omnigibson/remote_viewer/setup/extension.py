from __future__ import annotations

import asyncio
from pathlib import Path

import carb.settings
import carb.tokens
import omni.ext
import omni.kit.app
from omni.kit.mainwindow import get_main_window
from omni.kit.quicklayout import QuickLayout
from omni.kit.viewport.utility import get_viewport_from_window_name


class SetupExtension(omni.ext.IExt):
    """Configure layout and menu visibility for the remote viewer."""

    def on_startup(self, _ext_id: str) -> None:
        self._settings = carb.settings.get_settings()
        self._layout_task: asyncio.Task | None = asyncio.ensure_future(self._apply_layout())

        main_window = get_main_window()
        if main_window:
            main_window.get_main_menu_bar().visible = False

    def on_shutdown(self) -> None:
        if self._layout_task and not self._layout_task.done():
            self._layout_task.cancel()
        self._layout_task = None

    async def _apply_layout(self) -> None:
        app = omni.kit.app.get_app()
        for _ in range(4):
            await app.next_update_async()

        tokens = carb.tokens.get_tokens_interface()
        layout_root = Path(tokens.resolve("${omnigibson.remote_viewer.setup}/layouts"))
        layout_name = self._settings.get("/app/layout/name") or "default"
        layout_path = layout_root / f"{layout_name}.json"
        if layout_path.exists():
            QuickLayout.load_file(str(layout_path))
            viewport_api = get_viewport_from_window_name("Viewport")
            if viewport_api and hasattr(viewport_api, "fill_frame"):
                viewport_api.fill_frame = True
