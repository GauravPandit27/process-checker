# 🏭 Industrial Process Surveillance System

AI-powered computer vision application for monitoring industrial manufacturing processes.

## Architecture

```
Camera/Video → YOLO Detection → Object Tracking → Zone Assignment
→ Event Generation (debounced) → Process State Machine → Rule Validation
→ OK / NOT OK → Alarm Manager → Dashboard + SQLite Logging
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Upload a Video

Upload a factory video through the sidebar, or the default video will be detected automatically.

## Process Flow

```
IDLE → INPUT_DETECTED → PROCESSING → PROCESSING_COMPLETED → OUTPUT_CONFIRMED → COMPLETED → IDLE
```

### Valid Cycle (OK)
```
Operator picks part (Zone 1)
    → Part enters machine (Zone 2)
    → Processing completes within 15s
    → Part reaches output (Zone 3)
    → ✅ OK
```

### Violations (NOT OK)
- **PROCESSING_TIMEOUT**: Processing exceeds 15 seconds
- **PROCESS_SEQUENCE_VIOLATION**: Wrong order (e.g., INPUT → OUTPUT)
- **MISSING_PROCESS**: Part skips processing
- **UNEXPECTED_OUTPUT**: Output without prior input
- **INACTIVITY_TIMEOUT**: No valid input for 15 minutes

## Three Monitoring Zones

| Zone | Name | Purpose |
|------|------|---------|
| Zone 1 | Input / Picking | Operator picks parts from crate |
| Zone 2 | Processing | Machine processes the part |
| Zone 3 | Output / Conveyor | Processed part exits |

Zones are fully configurable via `config/zones.json` and the Streamlit sidebar.

## Project Structure

```
├── app.py                    # Main Streamlit dashboard
├── config/
│   ├── zones.json            # Zone polygon coordinates
│   └── settings.json         # Default thresholds & settings
├── vision/
│   ├── detector.py           # YOLO detection wrapper
│   ├── tracker.py            # Object tracking (ByteTrack)
│   └── zone_manager.py       # Zone polygon logic & overlays
├── process/
│   ├── state_machine.py      # Process state machine
│   ├── event_engine.py       # Debounced event generation
│   └── rules.py              # Timeout & sequence rules
├── alarm/
│   └── alarm_manager.py      # Alarm lifecycle management
├── database/
│   └── event_logger.py       # SQLite event logging
├── utils/
│   └── video_processor.py    # Video I/O & frame annotation
├── data/
│   ├── events/               # Event exports
│   ├── alarms/               # Alarm exports
│   └── snapshots/            # Violation evidence screenshots
└── requirements.txt
```

## Configuration

All thresholds are configurable via the Streamlit sidebar:

| Setting | Default | Description |
|---------|---------|-------------|
| Processing Timeout | 15 sec | Max processing time in Zone 2 |
| Inactivity Timeout | 15 min | Max time without valid input |
| Detection Confidence | 0.35 | YOLO confidence threshold |
| Debounce Frames | 5 | Frames needed to confirm detection |
| YOLO Model | yolov8n.pt | Detection model (n/s/m/l) |

## Key Design Principles

1. **YOLO detects objects** → our application understands the process
2. **Temporal debouncing** prevents false alarms from single frames
3. **Per-part state tracking** follows each object through the complete cycle
4. **Person presence ≠ process activity** — valid input requires part movement
5. **Modular architecture** — each component can be replaced independently
# process-checker
