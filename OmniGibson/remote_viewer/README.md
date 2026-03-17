# OmniGibson Remote Viewer

Browser-based streaming for OmniGibson simulations. Supports two streaming modes: **WebRTC** (interactive, high quality) and **WebSocket** (view-only, TCP-only).

## Quick Start

### 1. Setup

Run the setup script from this directory:

```bash
cd /path/to/BEHAVIOR-1K/OmniGibson/remote_viewer
./setup.sh
```

This installs Node.js via nvm (if needed) and downloads the NVIDIA WebRTC streaming library.

**Note:** NVIDIA library download requires valid registry credentials in `.npmrc` (WebRTC mode only). Node.js installation is non-root and per-user via nvm.

### 2. Choose Streaming Mode

Select the mode based on your network environment:

```bash
# WebRTC (Interactive): For local networks or cloud providers with UDP support
export OMNIGIBSON_REMOTE_STREAMING="webrtc"

# WebSocket (View-Only): For TCP-only environments or when UDP is blocked
export OMNIGIBSON_REMOTE_STREAMING="websocket"
```

### 3. Launch OmniGibson

Start any example:

```bash
python -m omnigibson.examples.scenes.scene_selector
```

The server starts on port 8211 by default.

### 4. Connect

Open your browser and navigate to the appropriate URL based on your deployment:

- **Local development**: `http://localhost:8211/`
- **Remote machine (direct IP)**: `http://<MACHINE_IP>:8211/`
- **Cloud with HTTPS proxy**: Use the URL provided by your provider (e.g., `https://abc123-8211.proxy.example.net`)
- **Cloud with port mapping**: `http://<EXTERNAL_IP>:<EXTERNAL_PORT>/`

The web UI automatically detects and uses the correct streaming mode.

## Streaming Modes

### WebRTC Mode (Interactive)

High-quality interactive streaming using WebRTC protocol.

**Best for:**
- Local development
- Cloud providers that expose UDP ports
- Scenarios requiring low latency and high quality

**Features:**
- H.264 video encoding (adaptive bitrate)
- Lower latency than WebSocket mode
- Full mouse and keyboard interaction
- 60 FPS streaming

**Requirements:**
- UDP port accessible (media stream)
- TCP ports accessible (signaling, HTTP)
- NVIDIA WebRTC streaming library

**Default ports:**
- HTTP server: 8211 (TCP)
- WebRTC signaling: 49100 (TCP)
- WebRTC media: 47998 (UDP)

### WebSocket Mode (View-Only)

TCP-only streaming for restrictive network environments.

**Best for:**
- Cloud providers with HTTP/HTTPS proxy only
- Environments where UDP is blocked
- Scenarios where viewing is sufficient

**Features:**
- Pure TCP transport (no UDP required)
- Works through any HTTP proxy or firewall
- MJPEG video streaming
- Approximately 30 FPS

**Limitations:**
- View-only (no mouse/keyboard interaction)
- Higher latency than WebRTC mode
- Medium quality (JPEG compression)
- No audio support

**Default ports:**
- HTTP server: 8211 (TCP, handles all traffic)

## Connection Instructions

### Local Development

No additional configuration needed:

```bash
export OMNIGIBSON_REMOTE_STREAMING="webrtc"  # or "websocket"
python -m omnigibson.examples.scenes.scene_selector
```

Access: `http://localhost:8211/`

### Remote Machine (Direct IP Access)

For machines accessible via direct IP:

```bash
export OMNIGIBSON_REMOTE_STREAMING="webrtc"  # or "websocket"
export PUBLIC_IP="<machine-public-ip>"
python -m omnigibson.examples.scenes.scene_selector
```

Access: `http://<machine-public-ip>:<http-port>/`

**Port mapping note:** If your ports are exposed 1:1 (external = internal), no additional configuration needed. Only set `EXTERNAL_*` variables if your provider remaps ports.

### Cloud Provider with HTTPS Proxy

Many cloud providers offer HTTPS proxies with automatic SSL termination and port mapping:

```bash
export OMNIGIBSON_REMOTE_STREAMING="websocket"  # Recommended for proxied environments
python -m omnigibson.examples.scenes.scene_selector
```

Access: Use the HTTPS URL provided by your provider

**Do not set `PUBLIC_IP`** - the system auto-detects the host from proxy headers.

### Cloud Provider with External Port Mapping

If your provider maps external ports to internal ones:

```bash
export OMNIGIBSON_REMOTE_STREAMING="websocket"  # Recommended for TCP-only
export PUBLIC_IP="<external-ip>"
export EXTERNAL_HTTP_PORT=<external-port>

# For WebRTC mode, also set:
export EXTERNAL_STREAM_SIGNALING_PORT=<external-signaling-port>
export EXTERNAL_STREAM_MEDIA_PORT=<external-media-port>

python -m omnigibson.examples.scenes.scene_selector
```

Access: `http://<external-ip>:<external-port>/`

## Environment Variables

### Required

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `OMNIGIBSON_REMOTE_STREAMING` | `"webrtc"`, `"websocket"`, `"native"` | `None` (disabled) | Streaming mode |

### Optional (Auto-detected)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PUBLIC_IP` | IP or hostname | Auto-detected | External IP address for clients |
| `OMNIGIBSON_HTTP_PORT` | Port number | `8211` | Internal HTTP server port |
| `OMNIGIBSON_WEBRTC_PORT` | Port number | `49100` | Internal WebRTC signaling port |

### WebRTC Mode Only

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EXTERNAL_STREAM_SIGNALING_PORT` | Port number | `49100` | External WebRTC signaling port |
| `EXTERNAL_STREAM_MEDIA_PORT` | Port number | `47998` | External WebRTC media port |

### WebSocket Mode Only

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `EXTERNAL_HTTP_PORT` | Port number | Auto-detected | External HTTP port (if different from internal) |

**Note:** Only set `EXTERNAL_*` variables when your cloud provider remaps ports. If ports are exposed 1:1, the defaults will work.

## Architecture

### System Overview

```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │
       ├─ HTTP GET /                    → Web UI (index.html)
       │
       ├─ POST /v1/streaming/session    → Session config (includes mode)
       │
       ├─ WebRTC Mode:
       │   ├─ WebSocket /sign_in        → Signaling (TCP)
       │   └─ UDP :47998                → Media stream (UDP)
       │
       └─ WebSocket Mode:
           └─ WebSocket /streaming/client/  → MJPEG frames (TCP)
```

### Extensions

The remote viewer consists of three Kit extensions:

1. **omnigibson.remote_viewer.setup**: Configures Kit UI (hides menu bar, applies viewport layout)
2. **omnigibson.remote_viewer.web**: Serves web UI and provides `/v1/streaming/session` API endpoint
3. **omnigibson.remote_viewer.websocket**: Provides TCP-only streaming (WebSocket mode only)

### How Mode Detection Works

The session endpoint (`/v1/streaming/session`) returns configuration including `streamingMode`:

```json
{
  "streamSignalingHost": "<server-ip>",
  "signalingPort": 8211,
  "mediaPort": 8211,
  "backendUrl": "http://<server-ip>:8211",
  "streamingMode": "websocket"
}
```

The web UI reads `streamingMode` and initializes the appropriate client:
- `"webrtc"`: NVIDIA AppStreamer library (WebRTC)
- `"websocket"`: Native WebSocket client (MJPEG over TCP)

## Performance Comparison

| Metric | WebRTC | WebSocket |
|--------|--------|-----------|
| Latency | Lower | Higher |
| Video Quality | High (H.264) | Medium (MJPEG) |
| Bandwidth | Variable (adaptive) | ~1-5 Mbps (depends on resolution) |
| CPU (Server) | Lower | Higher (JPEG encoding) |
| CPU (Client) | Low | Low |
| Firewall Compatibility | Medium (requires UDP) | High (TCP only) |
| Interaction | Full (mouse/keyboard) | View-only |

**Note:** Actual performance will vary based on network conditions, viewport resolution, scene complexity, and hardware.

## Troubleshooting

### Setup Issues

**"Failed to download NVIDIA library"**
- Verify `.npmrc` contains valid NVIDIA registry credentials
- Only required for WebRTC mode
- WebSocket mode works without this library

**"Node.js not found"**
- Run `./setup.sh` again - it will install Node.js via nvm automatically
- nvm installs Node.js per-user (no sudo required)
- If automatic installation fails, install manually: https://github.com/nvm-sh/nvm

### WebRTC Connection Issues

**Symptoms:**
- Browser connects to web UI but video doesn't appear
- Console shows "ICE connection failed" or "Media port unreachable"
- Connection attempts timeout

**Solutions:**
1. Verify UDP port 47998 is open and accessible from your network
2. Check that `EXTERNAL_STREAM_MEDIA_PORT` matches your actual external port mapping
3. Try WebSocket mode instead (TCP-only, no UDP required)

### WebSocket Connection Issues

**Symptoms:**
- Browser shows "WebSocket connection failed"
- HTTP requests work but WebSocket handshake fails
- 404 error on `/streaming/client/` endpoint

**Solutions:**
1. Verify `OMNIGIBSON_REMOTE_STREAMING="websocket"` is set before starting
2. Check server logs for extension loading errors
3. Ensure HTTP port is accessible from client network
4. Verify proxy/firewall allows WebSocket connections (not just HTTP)

### Web UI Shows "Setup Required"

**Symptoms:**
- Web UI displays error about missing setup
- NVIDIA streaming library not found
- Import errors in browser console

**Solution:**
```bash
cd /path/to/BEHAVIOR-1K/OmniGibson/remote_viewer
./setup.sh
```

Note: Only required for WebRTC mode.

### Session Endpoint Returns HTML

**Symptoms:**
- `/v1/streaming/session` returns HTML error page instead of JSON
- Browser console shows JSON parsing errors

**Solutions:**
1. Check server logs for extension startup errors
2. Verify extensions are enabled in simulator configuration
3. Restart the simulator

### High Latency or Stuttering

**Symptoms:**
- Video stream is choppy or delayed
- Frames are dropped

**Solutions:**
1. Reduce viewport resolution (WebSocket mode)
2. Check network bandwidth between client and server
3. Monitor server CPU usage (JPEG encoding is CPU-intensive in WebSocket mode)
4. Switch to WebRTC mode for lower latency (if UDP is available)

## Known Limitations

**WebSocket Mode:**
- View-only (no mouse/keyboard interaction)
- MJPEG compression quality lower than H.264
- Fixed frame rate (approximately 30 FPS)
- No audio streaming

**WebRTC Mode:**
- Requires UDP ports (not available in some cloud environments)
- More complex network configuration
- May require STUN/TURN servers for NAT traversal in some cases

## Technical Documentation

For more detailed technical information:

- **WebSocket extension**: See `kit/exts/omnigibson.remote_viewer.websocket/README.md`
- **Web UI implementation**: See `kit/exts/omnigibson.remote_viewer.web/web-ui/README.md`

## License

The NVIDIA Omniverse WebRTC Streaming Library is proprietary and cannot be redistributed. Each user must download it individually via the setup script.

