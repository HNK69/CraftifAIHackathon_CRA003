// shape_json.h -- writing the generic VMS overlay contract from a pipeline.
//
// The console (../../vms_app) renders TYPED SHAPES, not detections:
//
//   {"t":1785439154.540,"source":"cam0","frame":5,
//    "space":{"w":1280,"h":720},
//    "shapes":[
//      {"kind":"box","x":75.1,"y":317.9,"w":63.6,"h":193.8,
//       "label":"person","id":17,"score":0.4705,"text":"12.4 m"},
//      {"kind":"polygon","label":"drivable","points":[[0,700],[1280,700],[640,420]]},
//      {"kind":"polyline","label":"lane","points":[[10,720],[600,430]]},
//      {"kind":"skeleton","label":"person","points":[[..],[..]],"links":[[0,1]]}
//    ],
//    "fields":{"objects":7}}
//
// WHY SHAPES AND NOT `objects[]`. A VMS has to render whatever pipeline is in
// front of it -- boxes from a detector, lanes from a lane model, skeletons from
// pose, drivable area from a segmenter, a distance from LiDAR fusion. Bolting
// each onto a detection-shaped payload is how a VMS ends up welded to one
// pipeline. The console knows a handful of geometric primitives and nothing
// about what they mean, so a new pipeline needs no change to it.
//
// `t` is a WALL-CLOCK epoch timestamp and is the only field that can be joined
// against the recordings. Everything else is presentation.
//
// The console still reads the older detection-only `objects[]` payload (see
// ../../video_json_export and ../../mqtt_pipeline), so adopting this does not
// strand the pipelines writing that.

#ifndef SHAPE_JSON_H_
#define SHAPE_JSON_H_

#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsmeta.h"

namespace shape_json {

// A JSON string must escape quotes, backslashes and control characters, and
// labels come from a model's labels file -- not something to trust the shape of.
inline void escape(std::string& out, const char* s) {
    out += '"';
    for (const unsigned char* p = (const unsigned char*) s; p && *p; ++p) {
        switch (*p) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (*p < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", *p); out += b; }
                else out += (char) *p;
        }
    }
    out += '"';
}

inline void appendf(std::string& out, const char* fmt, ...)
    __attribute__((format(printf, 2, 3)));
inline void appendf(std::string& out, const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n > 0)
        out.append(buf, (size_t) (n < (int) sizeof(buf) ? n : sizeof(buf) - 1));
}

// ── one line, built incrementally ───────────────────────────────────────────
//
// Usage:
//     shape_json::Line line(now_epoch(), source_id, frame_num, W, H);
//     line.box(x, y, w, h, "person", id, 0.87, "12.4 m");
//     line.polygon(pts, n, "drivable");
//     line.field_i("objects", n);
//     fwrite(line.finish().data(), ...);
struct Line {
    std::string body;      // the shapes array contents
    std::string fields;    // the fields object contents
    double t;
    unsigned source_id, frame;
    int w, h;
    size_t n_shapes = 0;

    Line(double t_, unsigned source_id_, unsigned frame_, int w_, int h_)
        : t(t_), source_id(source_id_), frame(frame_), w(w_), h(h_) {
        body.reserve(512);
    }

    void sep() { if (n_shapes++) body += ", "; }

    // Common tail: label / id / score / text, all optional.
    void tail(const char* label, int64_t id, double score, const char* text) {
        if (label && *label) { body += ", \"label\": "; escape(body, label); }
        // -1 means "no tracker upstream, or an object it has not confirmed".
        // Emitting UNTRACKED_OBJECT_ID as-is would put 18446744073709551615 in
        // the payload, which most JSON readers silently turn into a float.
        if (id >= 0) appendf(body, ", \"id\": %" PRId64, id);
        if (score > 0.0) appendf(body, ", \"score\": %.4f", score);
        if (text && *text) { body += ", \"text\": "; escape(body, text); }
        body += '}';
    }

    void box(double x, double y, double bw, double bh, const char* label,
             int64_t id = -1, double score = 0.0, const char* text = nullptr) {
        sep();
        appendf(body, "{\"kind\": \"box\", \"x\": %.1f, \"y\": %.1f,"
                      " \"w\": %.1f, \"h\": %.1f", x, y, bw, bh);
        tail(label, id, score, text);
    }

    void path(const char* kind, const double* xy, size_t npts, const char* label,
              double score = 0.0, const char* text = nullptr) {
        sep();
        appendf(body, "{\"kind\": \"%s\", \"points\": [", kind);
        for (size_t i = 0; i < npts; ++i)
            appendf(body, "%s[%.1f, %.1f]", i ? ", " : "", xy[2 * i], xy[2 * i + 1]);
        body += ']';
        tail(label, -1, score, text);
    }

    void polyline(const double* xy, size_t npts, const char* label,
                  const char* text = nullptr) {
        path("polyline", xy, npts, label, 0.0, text);
    }
    void polygon(const double* xy, size_t npts, const char* label,
                 const char* text = nullptr) {
        path("polygon", xy, npts, label, 0.0, text);
    }

    // Points plus the edges joining them. `links` is 2*nlinks indices into the
    // point list; the console ignores out-of-range indices rather than crashing.
    void skeleton(const double* xy, size_t npts, const int* links, size_t nlinks,
                  const char* label, double score = 0.0) {
        sep();
        body += "{\"kind\": \"skeleton\", \"points\": [";
        for (size_t i = 0; i < npts; ++i)
            appendf(body, "%s[%.1f, %.1f]", i ? ", " : "", xy[2 * i], xy[2 * i + 1]);
        body += "], \"links\": [";
        for (size_t i = 0; i < nlinks; ++i)
            appendf(body, "%s[%d, %d]", i ? ", " : "", links[2 * i], links[2 * i + 1]);
        body += ']';
        tail(label, -1, score, nullptr);
    }

    // Free-form scalars. The console displays them and never interprets them,
    // which is what keeps them usable by a pipeline it has never seen.
    void field_d(const char* key, double v) {
        if (!fields.empty()) fields += ", ";
        escape(fields, key);
        appendf(fields, ": %.3f", v);
    }
    void field_i(const char* key, long long v) {
        if (!fields.empty()) fields += ", ";
        escape(fields, key);
        appendf(fields, ": %lld", v);
    }
    void field_s(const char* key, const char* v) {
        if (!fields.empty()) fields += ", ";
        escape(fields, key);
        fields += ": ";
        escape(fields, v);
    }

    bool empty() const { return n_shapes == 0; }

    std::string finish() const {
        std::string out;
        out.reserve(body.size() + fields.size() + 160);
        // The wall clock first, because it is the field everything else is
        // joined on. %.3f = millisecond resolution, finer than one frame at any
        // rate worth recording.
        appendf(out, "{\"t\": %.3f, \"source\": \"cam%u\", \"frame\": %u,"
                     " \"space\": {\"w\": %d, \"h\": %d}, \"shapes\": [",
                t, source_id, frame, w, h);
        out += body;
        out += ']';
        if (!fields.empty()) { out += ", \"fields\": {"; out += fields; out += '}'; }
        out += "}\n";
        return out;
    }
};

// Convenience: every NvDsObjectMeta on a frame as a box shape. A pipeline that
// only detects needs nothing else; one that fuses adds its own shapes alongside.
inline void boxes_from_meta(Line& line, NvDsObjectMeta* const* objs, size_t n,
                            const char* const* texts = nullptr) {
    for (size_t i = 0; i < n; ++i) {
        NvDsObjectMeta* ob = objs[i];
        if (!ob) continue;
        const NvOSD_RectParams& r = ob->rect_params;
        const int64_t id = (ob->object_id == UNTRACKED_OBJECT_ID)
                               ? -1 : (int64_t) ob->object_id;
        line.box(r.left, r.top, r.width, r.height, ob->obj_label, id,
                 ob->confidence, texts ? texts[i] : nullptr);
    }
}

}  // namespace shape_json

#endif  // SHAPE_JSON_H_
