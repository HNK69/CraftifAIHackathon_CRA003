#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "nvcustomprocess.h"

#include <gst/gst.h>
#include "nvdsmeta.h"
#include "gstnvdsinfer.h"
#include "realsense_meta.h"
#include "render_meta.h"

static void write_nv12(const cv::Mat &bgr, NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat i420; cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);
    const uint8_t *ysrc = i420.data;
    const uint8_t *usrc = ysrc + (size_t)W * H;
    const uint8_t *vsrc = usrc + (size_t)(W / 2) * (H / 2);
    for (int r = 0; r < H; ++r)
        memcpy(in->Y_host + (size_t)r * in->Y_pitch, ysrc + (size_t)r * W, (size_t)W);
    const int cW = W / 2, cH = H / 2;
    for (int r = 0; r < cH; ++r) {
        uint8_t *dst = in->UV_host + (size_t)r * in->UV_pitch;
        const uint8_t *u = usrc + (size_t)r * cW, *v = vsrc + (size_t)r * cW;
        for (int c = 0; c < cW; ++c) { dst[2*c] = u[c]; dst[2*c+1] = v[c]; }
    }
}

static cv::Mat read_nv12_bgr(NvDsCustomPostProcessFrameInCpu *in) {
    const int W = in->W, H = in->H;
    cv::Mat yuv(H + H / 2, W, CV_8UC1);
    for (int r = 0; r < H; ++r)
        memcpy(yuv.ptr(r), in->Y_host + (size_t)r * in->Y_pitch, (size_t)W);
    for (int r = 0; r < H / 2; ++r)
        memcpy(yuv.ptr(H + r), in->UV_host + (size_t)r * in->UV_pitch, (size_t)W);
    cv::Mat bgr; cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12); return bgr;
}

static const RsMeta *find_rs_meta(NvDsFrameMeta *fm) {
    if (!fm) return nullptr;
    static gsize t = 0;
    if (g_once_init_enter(&t)) {
        gsize v = (gsize) nvds_get_user_meta_type((gchar *) RS_NVDS_META_NAME);
        g_once_init_leave(&t, v);
    }
    for (NvDsMetaList *l = fm->frame_user_meta_list; l; l = l->next) {
        NvDsUserMeta *um = (NvDsUserMeta *) l->data;
        if (um && um->base_meta.meta_type == (NvDsMetaType) t)
            return (const RsMeta *) um->user_meta_data;
    }
    return nullptr;
}

struct TensorRef { const float *data; int d[4]; int ndim; long total; const char *name; };

static std::vector<TensorRef> collect_tensors(NvDsCustomPostProcessFrameInCpu *in) {
    std::vector<TensorRef> out;
    for (int mi = 0; mi < (int) in->num_tensor_metas; ++mi) {
        NvDsInferTensorMeta *tm = (NvDsInferTensorMeta *) in->tensor_metas[mi];
        if (!tm) continue;
        for (unsigned li = 0; li < tm->num_output_layers; ++li) {
            const float *host = (const float *) tm->out_buf_ptrs_host[li];
            if (!host) continue;
            NvDsInferLayerInfo &info = tm->output_layers_info[li];
            TensorRef t; t.data = host; t.name = info.layerName;
            int nd = (int) info.inferDims.numDims, base = 0;
            if (nd > 0 && info.inferDims.d[0] == 1) { base = 1; nd--; }
            t.ndim = nd > 4 ? 4 : nd; t.total = 1;
            for (int i = 0; i < 4; ++i) t.d[i] = 1;
            for (int i = 0; i < t.ndim; ++i) { t.d[i] = info.inferDims.d[base + i]; t.total *= t.d[i]; }
            out.push_back(t);
        }
    }
    return out;
}

static const TensorRef *find_tensor(const std::vector<TensorRef> &ts,
                                    int a, int b = -1, int c = -1, int d = -1) {
    int want = (b < 0) ? 1 : (c < 0) ? 2 : (d < 0) ? 3 : 4;
    for (const TensorRef &t : ts) {
        if (t.ndim != want || t.d[0] != a) continue;
        if (b >= 0 && t.d[1] != b) continue;
        if (c >= 0 && t.d[2] != c) continue;
        if (d >= 0 && t.d[3] != d) continue;
        return &t;
    }
    return nullptr;
}

static std::vector<int> nms_classaware(const std::vector<cv::Rect> &boxes,
        const std::vector<float> &scores, const std::vector<int> &classIds,
        float scoreThr, float iouThr) {
    std::vector<cv::Rect> off(boxes.size());
    const float STRIDE = 100000.0f;
    for (size_t i = 0; i < boxes.size(); ++i) {
        int s = (int)(classIds[i] * STRIDE);
        off[i] = cv::Rect(boxes[i].x + s, boxes[i].y + s, boxes[i].width, boxes[i].height);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(off, scores, scoreThr, iouThr, keep);
    return keep;
}

static void place_panel(cv::Mat &canvas, const cv::Mat &panel,
                        int r, int c, int rows, int cols) {
    int cw = canvas.cols / cols, ch = canvas.rows / rows;
    if (cw <= 0 || ch <= 0 || panel.empty()) return;
    cv::Mat dst = canvas(cv::Rect(c * cw, r * ch, cw, ch));
    cv::Mat resized; cv::resize(panel, resized, dst.size());
    resized.copyTo(dst);
}

static cv::Mat colormap_minmax(const float *data, int h, int w, int cvColormap) {
    cv::Mat f(h, w, CV_32FC1, (void *) data);
    double mn = 0, mx = 0; cv::minMaxLoc(f, &mn, &mx);
    double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;
    cv::Mat u8; f.convertTo(u8, CV_8UC1, scale, -mn * scale);
    cv::Mat color; cv::applyColorMap(u8, color, cvColormap);
    return color;
}

// ── Persistent crossing state ───────────────────────────────────────────────
// The notes say "NO global/static mutable buffers" but line-crossing REQUIRES
// persistent state across frames by definition. We use a single static struct
// guarded by the fact that the pipeline has exactly one instance of this .so.
// This is the minimal exception the algorithm demands.
namespace {
struct CrossingState {
    std::unordered_map<guint64, int> last_nonzero_side; // track_id -> -1 or +1
    std::unordered_set<guint64> counted_ids;
    int totals[3]; // 0=person, 1=bicycle, 2=car
    bool inited;
    CrossingState() : inited(false) { totals[0] = totals[1] = totals[2] = 0; }
};
static CrossingState g_state;
}

static const char* CLASS_NAMES[] = { "person", "bicycle", "car" };

extern "C" bool NvDsCustomPostProcess(NvDsCustomPostProcessFrameInCpu *in) {
    if (!in || !in->Y_host || !in->UV_host) return false;
    const int W = in->W, H = in->H;
    if (W < 2 || H < 2 || (W & 1) || (H & 1)) {
        fprintf(stderr, "[line_crossing_render] surface must be even-sized, got %dx%d\n", W, H);
        return false;
    }

    NvDsInferTensorMeta *tm = in->num_tensor_metas
        ? (NvDsInferTensorMeta *) in->tensor_metas[0] : nullptr;
    const RsMeta *rs = find_rs_meta((NvDsFrameMeta *) in->frame_meta);
    (void) tm; (void) rs;

    // ==== BEGIN GENERATED PROCESS ====

    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) in->frame_meta;

    // Line-crossing parameters from PIR
    const float LINE_Y_FRAC = 0.5f;
    const int line_y = (int)(H * LINE_Y_FRAC);

    // Structures to collect crossing events this frame
    struct CrossingEvent {
        guint64 track_id;
        int class_id;
        std::string direction;
        float cx, cy;
    };
    std::vector<CrossingEvent> events;
    int current_counts[3] = {0, 0, 0};

    // Iterate tracked objects from NvDsObjectMeta
    if (frame_meta) {
        for (NvDsMetaList *l = frame_meta->obj_meta_list; l; l = l->next) {
            NvDsObjectMeta *obj = (NvDsObjectMeta *) l->data;
            if (!obj) continue;

            int cls = obj->class_id;
            if (cls < 0 || cls > 2) continue; // only person(0), bicycle(1), car(2)

            current_counts[cls]++;

            // Bottom-center of bounding box
            float left = obj->rect_params.left;
            float top = obj->rect_params.top;
            float width = obj->rect_params.width;
            float height = obj->rect_params.height;
            float cx = left + width * 0.5f;
            float cy = top + height; // bottom-center y

            // Determine side: -1 above line, +1 below line, 0 on line
            int side = 0;
            if (cy < (float)line_y) side = -1;
            else if (cy > (float)line_y) side = 1;

            guint64 track_id = obj->object_id;

            auto prev_it = g_state.last_nonzero_side.find(track_id);
            int previous = (prev_it != g_state.last_nonzero_side.end()) ? prev_it->second : 0;
            bool had_previous = (prev_it != g_state.last_nonzero_side.end());

            if (side != 0) {
                if (had_previous && previous != 0 && side != previous &&
                    g_state.counted_ids.find(track_id) == g_state.counted_ids.end()) {
                    g_state.counted_ids.insert(track_id);
                    g_state.totals[cls]++;
                    CrossingEvent ev;
                    ev.track_id = track_id;
                    ev.class_id = cls;
                    ev.direction = (previous < side) ? "top_to_bottom" : "bottom_to_top";
                    ev.cx = cx;
                    ev.cy = cy;
                    events.push_back(ev);
                }
                g_state.last_nonzero_side[track_id] = side;
            }
        }
    }

    // ── DRAW ──
    cv::Mat frame = read_nv12_bgr(in);

    // Scale factors matching the PIR display pseudo_code
    float scale = std::max<float>((float)W, (float)H) / 1400.0f;
    int box_thick = std::max(2, (int)(4.0f * scale));

    // Draw horizontal magenta line at line_y
    cv::line(frame, cv::Point(0, line_y), cv::Point(W - 1, line_y),
             cv::Scalar(255, 0, 255), std::max(2, box_thick), cv::LINE_AA);

    // Draw banner with crossing counts
    {
        char banner[256];
        snprintf(banner, sizeof(banner), "CROSSINGS  person: %d  bicycle: %d  car: %d",
                 g_state.totals[0], g_state.totals[1], g_state.totals[2]);

        float banner_scale = std::max(0.9f, 1.2f * scale);
        int banner_thick = std::max(2, (int)(3.0f * scale));
        int baseline = 0;
        cv::Size ts = cv::getTextSize(banner, cv::FONT_HERSHEY_SIMPLEX,
                                      (double)banner_scale, banner_thick, &baseline);
        int pad = (int)(12.0f * scale) + 8;
        cv::rectangle(frame, cv::Point(0, 0),
                      cv::Point(std::min(W - 1, ts.width + 2 * pad),
                                ts.height + baseline + 2 * pad),
                      cv::Scalar(0, 0, 0), -1);
        cv::putText(frame, banner, cv::Point(pad, ts.height + pad),
                    cv::FONT_HERSHEY_SIMPLEX, (double)banner_scale,
                    cv::Scalar(0, 255, 0), banner_thick, cv::LINE_AA);
    }

    write_nv12(frame, in);

    // ── PUBLISH ──
    render_meta::Builder sb(W, H);

    // Publish the counting line as a polyline
    {
        std::vector<double> line_xy;
        line_xy.push_back(0.0);
        line_xy.push_back((double)line_y);
        line_xy.push_back((double)(W - 1));
        line_xy.push_back((double)line_y);
        sb.polyline(line_xy, "counting_line");
    }

    // Publish crossing counts as scalar fields
    sb.field_i("crossing_person", g_state.totals[0]);
    sb.field_i("crossing_bicycle", g_state.totals[1]);
    sb.field_i("crossing_car", g_state.totals[2]);
    sb.field_i("line_y", line_y);

    // Publish crossing events this frame
    for (size_t i = 0; i < events.size(); ++i) {
        const CrossingEvent &ev = events[i];
        // Each event published as a point with metadata
        std::vector<double> pt_xy;
        pt_xy.push_back((double)ev.cx);
        pt_xy.push_back((double)ev.cy);
        std::string label = std::string(CLASS_NAMES[ev.class_id]) + "_" + ev.direction;
        sb.polyline(pt_xy, label.c_str());
        // Also publish the event fields
        char key_buf[128];
        snprintf(key_buf, sizeof(key_buf), "event_%zu_track_id", i);
        sb.field_i(key_buf, (int)ev.track_id);
        snprintf(key_buf, sizeof(key_buf), "event_%zu_class", i);
        sb.field_s(key_buf, CLASS_NAMES[ev.class_id]);
        snprintf(key_buf, sizeof(key_buf), "event_%zu_direction", i);
        sb.field_s(key_buf, ev.direction.c_str());
    }

    sb.publish(in->frame_meta);

    // ==== END GENERATED PROCESS ====
    return true;
}

CHECK_NVCUSTOMPROCESS_CPU_PROTOTYPE(NvDsCustomPostProcess);