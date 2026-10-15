CUDA_VER ?= 12.6
DS_VER   ?= 7.1
CXX      := g++
DS_INC   := /opt/nvidia/deepstream/deepstream-$(DS_VER)/sources/includes
DS_LIB   := /opt/nvidia/deepstream/deepstream-$(DS_VER)/lib
# nvinfer_backport headers (classifier-with-labels signature). Override if
# your backport install puts them elsewhere; defaults to DeepStream includes.
BACKPORT_INC ?= $(DS_INC)
OUT_DIR  ?= ../build

# nvinfer custom-parser flags.
CFLAGS := -fPIC -O2 -std=c++14 \
  -I /usr/local/cuda-$(CUDA_VER)/include \
  -I $(DS_INC) \
  -I $(BACKPORT_INC) \
  $(shell pkg-config --cflags opencv4)

# nvcustomprocess CPU/OpenCV user-library flags. nvcustomprocess.h ships
# in $(DS_INC); realsense_meta.h installs to /usr/local/include. No CUDA.
# LINK: always -lnvds_meta (nvds_get_user_meta_type lives there) and
# -Wl,--no-undefined (turns any missing symbol into a BUILD error instead
# of a silent runtime dlopen failure / stale-.so trap). The --libs package
# list MUST match --cflags (glib-2.0 gstreamer-1.0 opencv4) — the parser
# uses glib (g_once_init / g_quark via the RsMeta lookup), so linking only
# opencv leaves those undefined and --no-undefined fails the build.
NVCP_PKGS := glib-2.0 gstreamer-1.0 opencv4
NVCP_CFLAGS := -fPIC -O2 -std=c++17 \
  -I $(DS_INC) \
  -I /usr/local/include \
  $(shell pkg-config --cflags $(NVCP_PKGS))
NVCP_LDFLAGS := -shared -Wl,--no-undefined \
  -L $(DS_LIB) -lnvds_meta -Wl,-rpath,$(DS_LIB) \
  $(shell pkg-config --libs $(NVCP_PKGS))

# MQTT publisher flags. libmosquitto ships in the DeepStream image under
# /usr/local, which is a SEPARATE pkg-config search path from the system
# one — querying it without PKG_CONFIG_PATH silently yields empty flags
# and the link then fails on mosquitto_lib_init. No OpenCV and no CUDA:
# this library reads metadata only and never touches a pixel.
# (Do NOT link the copy under /opt/tritonserver/.../mosquitto-2.0.15 —
# that is a leftover build tree with no .so symlink and no pkg-config.)
MOSQ_PC     := PKG_CONFIG_PATH=/usr/local/lib/pkgconfig pkg-config
MQTT_PKGS   := glib-2.0
MQTT_CFLAGS := -fPIC -O2 -std=c++14 -Wall \
  -I $(DS_INC) \
  -I /usr/local/include \
  $(shell pkg-config --cflags $(MQTT_PKGS)) \
  $(shell $(MOSQ_PC) --cflags libmosquitto)
MQTT_LDFLAGS := -shared -Wl,--no-undefined \
  -L $(DS_LIB) -lnvds_meta -Wl,-rpath,$(DS_LIB) \
  -Wl,-rpath,/usr/local/lib \
  $(shell pkg-config --libs $(MQTT_PKGS)) \
  $(shell $(MOSQ_PC) --libs libmosquitto)

# JSON file-writer flags — the mqtt flags without mosquitto. It writes to
# a FILE and never touches a pixel, so no network library, no OpenCV,
# no CUDA.
JSON_CFLAGS  := -fPIC -O2 -std=c++14 -Wall \
  -I $(DS_INC) \
  -I /usr/local/include \
  $(shell pkg-config --cflags glib-2.0)
JSON_LDFLAGS := -shared -Wl,--no-undefined \
  -L $(DS_LIB) -lnvds_meta -Wl,-rpath,$(DS_LIB) \
  $(shell pkg-config --libs glib-2.0)

all: $(OUT_DIR)/libconfig_det_pgie_parser.so $(OUT_DIR)/libline_crossing_render_process.so $(OUT_DIR)/libmeta_json.so $(OUT_DIR)/libvms_detect.so

$(OUT_DIR)/libconfig_det_pgie_parser.so: config_det_pgie_parser.cpp Makefile
	mkdir -p $(OUT_DIR)
	rm -f $@
	$(CXX) $(CFLAGS) -o $@ config_det_pgie_parser.cpp -shared -Wl,--no-undefined

$(OUT_DIR)/libline_crossing_render_process.so: line_crossing_render_process.cpp render_meta.h Makefile
	mkdir -p $(OUT_DIR)
	rm -f $@
	$(CXX) $(NVCP_CFLAGS) -o $@ line_crossing_render_process.cpp $(NVCP_LDFLAGS)

$(OUT_DIR)/libmeta_json.so: meta_json.cpp frame_json.h render_meta.h Makefile
	mkdir -p $(OUT_DIR)
	rm -f $@
	$(CXX) $(JSON_CFLAGS) -o $@ meta_json.cpp $(JSON_LDFLAGS)

$(OUT_DIR)/libvms_detect.so: vms_detect.cpp shape_json.h event_sink.h Makefile
	mkdir -p $(OUT_DIR)
	rm -f $@
	$(CXX) $(JSON_CFLAGS) -o $@ vms_detect.cpp $(JSON_LDFLAGS)

clean:
	rm -f $(OUT_DIR)/libconfig_det_pgie_parser.so $(OUT_DIR)/libline_crossing_render_process.so $(OUT_DIR)/libmeta_json.so $(OUT_DIR)/libvms_detect.so
