# FaceX on ESP32-P4

Espressif's RISC-V MCU (dual HP RV32 @ 360 MHz, 768 KB SRAM, up to 32 MB
PSRAM, MIPI-CSI camera input, no NPU). FaceX ships an ESP-IDF component
(`components/facex/`) and a runnable camera example (`examples/esp32p4_camera/`).

## Status, plainly

| Piece | Today | Path to production |
|---|---|---|
| MIPI-CSI capture (SC2336 / OV5647 / …) | ✅ working | use as-is |
| Camera → downscale → FaceX dispatch | ✅ working | use as-is |
| Stub face detector (synthetic boxes) | ✅ working | for bring-up only |
| Native FaceX engine (EdgeFace-XS) | ⚠️ compiles, ~1-3 s/frame | not for shipping |
| Distilled EdgeFace-Nano + ESP-NN backend | ❌ not built | future work |
| PPA-accelerated downscale | ❌ scalar NN | future work |

The camera bridge is real and complete. The model story is the next
multi-week of work — distilled EdgeFace-Nano + an ESP-NN backend.
See the "Roadmap" section at the bottom of this file.

## Prerequisites

- **ESP-IDF v5.4 or newer** (the camera_driver API and the
  `espressif/esp_cam_sensor` component arrived in v5.4 stable).
  ```bash
  git clone -b v5.4 --depth 1 --recurse-submodules https://github.com/espressif/esp-idf.git ~/esp-idf
  ~/esp-idf/install.sh esp32p4
  . ~/esp-idf/export.sh
  ```
- **Hardware:** ESP32-P4-Function-EV-Board with the SC2336 module
  pre-installed, or any other sensor supported by `esp_cam_sensor`.
  Pin assignments and LDO channel default to the EV-Board layout.
- **PSRAM:** required. Frame buffers (≈1 MB at 800×640 RGB565 ×2)
  cannot fit in the 768 KB internal SRAM.

## Build, flash, run

```bash
cd examples/esp32p4_camera
idf.py set-target esp32p4
idf.py menuconfig          # optional: sensor / GPIOs / FaceX backend
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

Expected first-run console (stub backend, no real face anywhere):

```
I app: FaceX ESP32-P4 MIPI-CSI camera example starting
I app: sensor detected: SC2336
I app: FaceX ready, backend=stub, detector input=96x96
I app: init complete; capture task running on core 1
I app: 28.7 fps, last detect=42 us, last n_faces=1, backend=stub
```

If no `28.7 fps` line appears within ~3 s of the boot banner, see
"Troubleshooting" below.

## How the example wires everything

Following the [ESP-IDF camera_driver](https://docs.espressif.com/projects/esp-idf/en/stable/esp32p4/api-reference/peripherals/camera_driver.html)
recipe verbatim:

```
LDO 2.5 V on LDO_VO3 → enables CSI PHY rail
        ↓
SCCB I2C bus (port 0, GPIO 7/8 by default)
        ↓
esp_cam_sensor_detect()       — auto-detects SC2336 / OV5647 / etc.
        ↓
esp_cam_sensor_set_format()   — picks 800x640 + 2 lanes + RGB565
        ↓
esp_cam_new_csi_ctlr(...)     — CSI controller handle
        ↓
register on_get_new_trans / on_trans_finished callbacks
        ↓
allocate N PSRAM frame buffers (DMA-cap, 64-byte aligned)
        ↓
esp_cam_ctlr_enable + start
        ↓
loop: esp_cam_ctlr_receive(blocking) → downscale → facex_esp_detect → re-receive
```

Source: `examples/esp32p4_camera/main/app_main.c`.

## FaceX backend selection

`idf.py menuconfig` → **FaceX** → **Inference backend**:

- **Stub** (default) — synthetic deterministic face, useful for proving
  the camera + UI plumbing without touching neural weights.
- **Native FaceX engine** — links `src/edgeface_engine.c` and friends.
  Compiles, but EdgeFace-XS is too large for real-time on a 360 MHz
  RV32 — expect 1-3 s per frame. Provided as an evaluation crutch.
  Requires you to define `facex_esp_native_weights_path()` returning
  a `fopen`-able path (typically an SD-card mount or an `EMBED_TXTFILES`
  build artefact).
- **ESP-NN** — reserved Kconfig slot. Sprint C5 will fill this in with
  a backend that dispatches each conv via Espressif's PIE-SIMD INT8
  kernels. Requires the EdgeFace-Nano distilled model from sprint C1.

## Resource budget (target SoC = ESP32-P4)

For the **default stub** backend on the SC2336 + 800×640 RGB565 path:

| Resource | Used | Available |
|---|---:|---:|
| Internal SRAM (DRAM) | ≈ 80 KB | 768 KB |
| PSRAM | ≈ 2.1 MB (2× frame, 1× detect) | up to 32 MB |
| Flash | ≈ 600 KB (idf-bootloader + app) | typically 16 MB |
| CPU on capture core | ≈ 6 % | 360 MHz HP RV32 |
| FPS | ≈ 28-30 | sensor-limited |

The native backend pushes PSRAM to ~10 MB and CPU to 100 % at 0.5 fps —
useful for a one-off "yes the engine compiled" check, not for product.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `no MIPI sensor responded on SCCB` | Wrong I2C pins, or sensor not powered | Check `CONFIG_SCCB_SCL_GPIO` / `SDA_GPIO`; on EV-Board they default to GPIO 8 / 7. Verify camera ribbon orientation. |
| `frame[0] alloc … failed (PSRAM exhausted?)` | PSRAM not enabled in sdkconfig | `idf.py menuconfig` → Component config → ESP PSRAM → enable. The example's `sdkconfig.defaults` does this — make sure you didn't override it. |
| Bootloop right after `Camera ready` | LDO not delivering 2.5 V | Confirm `LDO_MIPI_PHY_CHAN` matches your board. On EV-Board it's channel 3 / `LDO_VO3`. |
| `0.0 fps` lines | Callbacks not firing → frames never finish | Check sensor format match (resolution + lane count + bit-rate) — `esp_cam_sensor_query_format` lists what the sensor actually supports. |
| App link fails with `undefined reference to facex_esp_native_weights_path` | You enabled the Native backend but didn't provide the weights-path callback | Either implement that function in your application code, or switch back to the Stub backend until the EdgeFace-Nano sprint lands. |

## Files

```
components/facex/
  Kconfig                       — backend selection menu
  CMakeLists.txt                — pulls in src/ when native is selected
  include/facex_esp.h           — public API: init / detect / free
  src/facex_esp.c               — backend dispatch (stub + native)

examples/esp32p4_camera/
  CMakeLists.txt                — top-level IDF project
  sdkconfig.defaults            — PSRAM on, CPU @ 360 MHz, etc.
  main/CMakeLists.txt           — component requires + sources
  main/Kconfig.projbuild        — sensor / pin / lane config
  main/idf_component.yml        — pulls esp_cam_sensor
  main/app_main.c               — full CSI bring-up + capture task
  README.md                     — short start-here for the example
```

## Roadmap

The camera bridge ships now. The remaining work that turns this into
a shipping face-recognition product on ESP32-P4:

- Distill EdgeFace-Nano (target: ~300 K params, 64×64 input, 256-d
  embedding, no XCA attention) — fits in PSRAM, can run real-time on P4.
- Distill YuNet-Mini (~50 K params, 96×96, 8-bit) — replaces the
  current detector at the edge of the size budget.
- `src/backend_espnn.c` — dispatch convs through Espressif's `esp-nn`
  PIE-SIMD INT8 kernels. Required for both detector and embedder.
- PSRAM streaming weights + cache prefetch — the engine currently
  loads weights eagerly; for production-fit models we need streaming.
- EV-Board demo wiring — replace `facex_esp_detect` stub call with
  the real ESPNN backend once it's ready.
- Power profiling on the EV-Board.

Until the EdgeFace-Nano work lands the example is best understood as
a "MIPI-CSI capture loop with a face-detector seam". The seam is the
API the rest of the work fills in. See `docs/implementation.md` §4 for
the implementation snapshot.
