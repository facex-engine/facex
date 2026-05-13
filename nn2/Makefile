CC      = /c/mingw64/bin/gcc
CFLAGS  = -O3 -march=native -funroll-loops -ffast-math -fomit-frame-pointer \
          -fno-plt -fvisibility=hidden -DNDEBUG \
          -Wno-unused-function -Wno-unused-variable -Iinclude -Isrc
LDFLAGS = -lm -lsynchronization
LTO     = -flto

# PGO targets (10-18% free speedup)
PGO_DIR = pgo_data

CORE_SRC = src/gemm.c src/gemm_res.c src/conv.c src/conv_implicit.c src/conv_fused.c \
           src/conv_tiled.c src/winograd.c \
           src/ops.c src/decode.c src/pool.c src/net.c src/nn2.c
CORE_OBJ = $(CORE_SRC:.c=.o)

INT8_SRC = src/gemm_int8.c
INT8_OBJ = $(INT8_SRC:.c=.o)

NVR_SRC = src/nvr.c
NVR_OBJ = $(NVR_SRC:.c=.o)

ND_SRC = src/nanodet.c
ND_OBJ = $(ND_SRC:.c=.o)

ALL_OBJ = $(CORE_OBJ) $(INT8_OBJ)

all: nn2 nn2_img nn2_nvr nanodet

nn2: $(ALL_OBJ) src/main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

nn2_img: $(ALL_OBJ) src/main_img.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

nn2_nvr: $(ALL_OBJ) $(NVR_OBJ) src/main_nvr.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

nanodet: $(ALL_OBJ) $(ND_OBJ) src/main_nanodet.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f src/*.o nn2 nn2.exe nn2_img nn2_img.exe nn2_nvr nn2_nvr.exe nanodet nanodet.exe

# Export weights (requires Python + ultralytics)
export-320:
	python tools/export_yolov8n.py --size 320 --output weights/yolov8n_320.bin

export-640:
	python tools/export_yolov8n.py --size 640 --output weights/yolov8n.bin

export-nanodet:
	python tools/export_nanodet.py

bench: nn2
	./nn2 weights/yolov8n_320.bin test_image.raw 640 480 6

# LTO build (cross-module inline, dead code elimination)
lto: CFLAGS += $(LTO)
lto: LDFLAGS += $(LTO)
lto: all

# PGO Step 1: generate profile
pgo-gen: CFLAGS += -fprofile-generate=$(PGO_DIR)
pgo-gen: LDFLAGS += -fprofile-generate=$(PGO_DIR)
pgo-gen: clean all
	@echo "Run: ./nn2_img weights/yolov8n_320.bin test_image.jpg 6"
	@echo "Then: make pgo-use"

# PGO Step 2: use profile (combine with LTO for max effect)
pgo-use: CFLAGS += -fprofile-use=$(PGO_DIR) $(LTO) -fprofile-correction
pgo-use: LDFLAGS += -fprofile-use=$(PGO_DIR) $(LTO)
pgo-use: clean all
	@echo "Built with PGO+LTO — expected 10-18% faster"

.PHONY: all clean export-320 export-640 export-nanodet bench lto pgo-gen pgo-use
