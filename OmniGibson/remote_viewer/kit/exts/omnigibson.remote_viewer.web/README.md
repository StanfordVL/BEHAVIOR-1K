# OmniGibson Remote Viewer Web Extension

This extension serves the remote viewer web UI and provides the session configuration API via Kit's HTTP server.

## Components

### Static File Serving

Serves the vanilla JavaScript web UI from the `web-ui/` directory. The web UI provides:
- WebRTC streaming client (interactive, requires UDP)
- WebSocket streaming client (view-only, TCP-only)
- Automatic mode detection based on backend configuration

### Session Configuration API

Provides a `/v1/streaming/session` API endpoint for configuring remote streaming connections.

**Note:** This is **not** a full session management system like NVIDIA's `omni.kit.livestream.session` extension.

#### Context

NVIDIA's official session management (`omni.kit.livestream.session`) branched out from `omni.kit.livestream.nvcf` after v8.0.0. However, Kit 106.5 (used by Isaac Sim) only supports `omni.kit.livestream.nvcf` up to v6.2.0. While NVIDIA exposes streaming endpoints, there was no simple way to configure connection parameters for self-hosted deployments.

This extension fills that gap by:
- Reading environment variables for external IPs and ports
- Detecting streaming mode (WebRTC vs WebSocket)
- Auto-detecting host from request headers (e.g., behind HTTPS proxies)
- Returning connection configuration to the web UI

The `sessionId` field is not used for actual session tracking—this endpoint simply provides connection details to the client.

#### API Endpoint

**Endpoint:** `POST /v1/streaming/session`

**Response:**
```json
{
  "streamSignalingHost": "<host>",
  "signalingPort": 8211,
  "mediaPort": 8211,
  "backendUrl": "http://<host>:8211",
  "streamingMode": "websocket"
}
```

#### Environment Variables

- `PUBLIC_IP`: External IP address (optional, auto-detected from request headers if not set)
- `OMNIGIBSON_REMOTE_STREAMING`: Set to `"webrtc"` or `"websocket"`
- `EXTERNAL_HTTP_PORT`: External HTTP port (for WebSocket mode)
- `EXTERNAL_STREAM_SIGNALING_PORT`: External WebRTC signaling port
- `EXTERNAL_STREAM_MEDIA_PORT`: External WebRTC media port

## Web UI Setup

The NVIDIA Omniverse WebRTC Streaming Library must be downloaded separately due to licensing restrictions.

Run the setup script from the repository root:
```bash
cd /path/to/BEHAVIOR-1K/OmniGibson/remote_viewer
./setup.sh
```

This will:
1. Install Node.js via nvm if not present
2. Download the NVIDIA WebRTC streaming library from their private npm registry

## Directory Structure

```
omnigibson.remote_viewer.web/
├── README.md                      # This file
├── config/
│   └── extension.toml             # Extension configuration
├── omnigibson/
│   └── remote_viewer/
│       └── web/
│           └── extension.py       # Extension implementation
└── web-ui/
    ├── .npmrc                     # NVIDIA npm registry config
    ├── package.json               # npm dependencies
    ├── index.html                 # Single-file web UI
    └── node_modules/              # Downloaded libraries (gitignored)
```
