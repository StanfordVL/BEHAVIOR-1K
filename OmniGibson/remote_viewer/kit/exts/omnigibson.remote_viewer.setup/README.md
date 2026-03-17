# OmniGibson Remote Viewer Setup Extension

This extension configures Kit's UI for remote streaming by hiding unnecessary UI elements and applying a custom layout.

## Purpose

When streaming OmniGibson remotely, the default Kit UI (menu bars, toolbars, property panels) is not needed and can clutter the viewport. This extension:

1. **Hides the main menu bar** - Users interact through the web UI, not Kit's native menus
2. **Applies a custom layout** - Maximizes the viewport for streaming by removing side panels and docked windows

## How It Works

On startup, the extension:
- Hides Kit's main menu bar via `main_window.get_main_menu_bar().visible = False`
- Waits for Kit to fully initialize (4 frames)
- Loads a layout file from the `layouts/` directory
- Sets the viewport to fill the entire frame

## Layout Files

Custom layouts can be placed in:
```
omnigibson.remote_viewer.setup/layouts/
└── default.json
```

The layout name is determined by Kit's `/app/layout/name` setting.

## When to Use

This extension should be loaded whenever remote streaming is enabled (WebRTC or WebSocket mode) to provide a clean, distraction-free streaming experience.
