# FaceX × ESP32-P4 MIPI-CSI camera example

Captures from a MIPI camera, downscales each frame, hands it to the
FaceX detection wrapper, and logs FPS + bbox to UART.

## Hardware

- **Board:** ESP32-P4-Function-EV-Board
- **Sensor:** SC2336 (bundled). Other sensors supported by Espressif's
  `esp_cam_sensor` framework (OV5645, OV5647, etc.) work — adjust
  `CONFIG_CAM_*` and the SCCB pins via `idf.py menuconfig`.
- **Power:** the CSI PHY needs 2.5 V; the example acquires LDO channel
  3 at 2500 mV (`LDO_VO3` on the EV board). If your board routes a
  different LDO, change `LDO_MIPI_PHY_CHAN` in `main/app_main.c`.

## Build / flash

```bash
. $IDF_PATH/export.sh                  # ESP-IDF v5.4 or newer
cd examples/esp32p4_camera
idf.py set-target esp32p4
idf.py menuconfig                      # optional — sensor / GPIOs / FaceX backend
idf.py build flash monitor             # flashes and tails the UART
```

Expected console output (stub backend, default):

```
I (NNN) app: FaceX ESP32-P4 MIPI-CSI camera example starting
I (NNN) app: sensor detected: SC2336
I (NNN) facex: backend: stub (96x96, threshold=0.50)
I (NNN) app: FaceX ready, backend=stub, detector input=96x96
I (NNN) app: init complete; capture task running on core 1
I (NNN) app: 28.7 fps, last detect=42 us, last n_faces=1, backend=stub
```

## Backend selection (`idf.py menuconfig` → FaceX → Inference backend)

| Backend | Status | Latency / frame on P4 | Notes |
|---|---|---|---|
| **stub** (default) | Works | <100 µs | Synthetic face. Use for board bring-up, UI plumbing. |
| **native** | Compiles, very slow | 1-3 s | Loads the full EdgeFace-XS engine. Needs PSRAM, weights file path provided by `facex_esp_native_weights_path()`. Not for shipping. |
| **espnn** | Reserved | — | Future — distilled EdgeFace-Nano + ESP-NN backend. See `../../docs/esp32p4.md`. |

## What's wired vs. deferred

This example is the **camera-to-FaceX bridge**, complete and runnable.
What it does **not** ship:

- A production face-recognition model that fits ESP32-P4 (target:
  EdgeFace-Nano, ~300 K params, 64×64 input, 256-d embedding).
- An ESP-NN backend for the FaceX engine.
- PPA-accelerated downscale — the example uses a scalar nearest-neighbour.

The bridge code (`components/facex/src/facex_esp.c`) is the seam where
those land. Once they exist, switch the Kconfig backend, drop the
`.tflite` / weight artefact in via `idf.py add-dependency`, and rebuild.

## See also

- `../../docs/esp32p4.md` — fuller ESP32-P4 build guide and roadmap
- `../../docs/implementation.md` §4 — implementation snapshot of this
  ESP-IDF component, including the assumptions baked into the stubbed
  backend
- `../../components/facex/Kconfig` — backend selection options
- [ESP-IDF camera_driver doc](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/peripherals/camera_driver.html)
