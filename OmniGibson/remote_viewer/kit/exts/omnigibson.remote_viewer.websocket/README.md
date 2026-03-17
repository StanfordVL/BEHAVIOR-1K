# OmniGibson WebSocket Streaming Extension

TCP-only streaming for cloud environments that don't expose UDP ports.

**For general setup and usage instructions, see the main [Remote Viewer README](../../../README.md).**

This document covers extension-specific technical details.

## Quick Reference

```bash
export OMNIGIBSON_REMOTE_STREAMING="websocket"
python -m omnigibson.examples.scenes.scene_selector
```

Access: `http://localhost:8211/` (or appropriate remote URL)

## Overview

This extension captures viewport frames and streams them as MJPEG over WebSocket. Unlike WebRTC, it only requires TCP and works in any environment where HTTP/WebSocket is available.

## Features

- **Pure TCP**: No UDP required
- **WebSocket-based**: Works through any HTTP proxy or firewall
- **MJPEG streaming**: Simple, reliable video encoding
- **View-only mode**: Optimized for remote viewing (no interactive control)
- **Browser-compatible**: No special client app needed

## Comparison with WebRTC Mode

| Feature | WebRTC | WebSocket |
|---------|--------|-----------|
| Transport | UDP + TCP | TCP only |
| Latency | Lower | Higher |
| Quality | High (H.264) | Medium (MJPEG) |
| Firewall/Proxy | Requires UDP ports | Works everywhere |
| Interaction | Full mouse/keyboard | View-only |
| Cloud compatibility | UDP-enabled providers | All providers |

## Technical Details

### How It Works

1. Extension registers `/streaming/client/` WebSocket endpoint on Kit's HTTP server
2. Browser connects and receives JPEG frames at approximately 30 FPS
3. Frames are captured from the viewport using Kit's capture API
4. Each frame is compressed as JPEG and sent as a binary WebSocket message
5. Browser decodes and displays frames using canvas rendering

### Performance

- **Resolution**: Adaptive (matches viewport resolution)
- **Frame Rate**: Target 30 FPS (adjusts based on network conditions)
- **Bandwidth**: Approximately 1-5 Mbps (varies with scene complexity and resolution)
- **Latency**: Higher than WebRTC mode (exact value depends on network and encoding overhead)

### Limitations

- Lower quality than WebRTC due to JPEG compression artifacts
- Higher latency than WebRTC
- Higher CPU usage on server for JPEG encoding
- No audio support
- View-only mode (no interactive control)

## When to Use

**Use WebSocket mode when:**
- Deploying to cloud providers that only expose HTTP/HTTPS
- UDP ports are blocked by firewall
- WebRTC connection fails
- Only viewing is needed (no interaction required)

**Use WebRTC mode when:**
- UDP ports are available
- Lower latency is critical
- Higher quality is needed
- Interactive control (mouse/keyboard) is required

## Environment Variables

See the [main README](../../../README.md#environment-variables) for full configuration options.

Key variables for WebSocket mode:
- `OMNIGIBSON_REMOTE_STREAMING="websocket"`
- `OMNIGIBSON_HTTP_PORT` (default: 8211)
- `PUBLIC_IP` (optional, auto-detected)
- `EXTERNAL_HTTP_PORT` (optional, auto-detected)
