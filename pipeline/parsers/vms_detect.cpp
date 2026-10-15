// vms_events.cpp -- the VMS event log, for the `nvcustomprocess` plugin.
//
// One JSON line per camera per logged frame, appended to
// <VMS_ROOT>/events/cam<source_id>.jsonl. Entry point: VmsDetect.
//
// Emits the GENERIC overlay contract (shape_json.h): a list of typed shapes in
// a stated coordinate space, not a detection-shaped `objects[]`. A detector only
// ever produces boxes, so this file is short -- but it writes them in the form
// the console renders for every pipeline, which is what lets ../fusion add
// polygons and distances beside them with no change to the console at all.
//
// AND THAT IS WHY THIS FILE ALSO FORWARDS render_meta SHAPES. shape_json.h has
// always been able to write a skeleton, a polygon and a polyline, and until now
// NOTHING EVER CALLED THOSE WRITERS -- a pose pipeline's console showed boxes
// from its detector and nothing of the pose, and a depth- or lane-only pipeline
// showed nothing at all, because the only code that knew the tensor layout was
// the generated render .so and it drew with OpenCV and dropped what it decoded.
// render_meta.h closes that: the render stage publishes ONE user meta per frame
// carrying shapes in EXACTLY this vocabulary, and this file hands them to the
// same writers. No new `kind` reaches the console, no key changes shape, and
// one producer feeds the console, the .json and the MQTT topic alike.
//
// A frame nobody published on appends NOTHING -- not an empty array, not a null
// -- so a detection pipeline's line is byte-for-byte what it was before any of
// this existed. That is a hard requirement: those lines are diffed byte for
// byte against a production log.
//
// AND IT ALSO READS classifier_meta_list, for the third family that puts its
// answer somewhere other than the object's box. A classification model
// (network-type=1) does not fill NvDsObjectMeta at all: nvinfer fabricates ONE
// whole-frame object per frame with class_id = -1, an EMPTY obj_label and an
// unassigned confidence, and hangs the actual result off
// obj->classifier_meta_list. Nothing here read it, so -- measured on the user's
// own EfficientNet-B2 project, 59 logged frames -- the console got
// {"shapes": [], "fields": {"objects": 0}} at the generated default
// VMS_MIN_CONF=0.3 and an unlabelled 1920x1080 rectangle at 0.0. Two separate
// defects, and BOTH are needed: the confidence gate below ORs on the presence
// of classifier metadata (the placeholder's confidence is 0.0f and no gate
// above zero can ever admit it), and the box loop turns the results into the
// label, score and text of the box that is already being written.
//
// WHAT MAKES THIS A VMS EVENT LOG RATHER THAN A DETECTION DUMP: every line
// carries `t`, a WALL-CLOCK epoch timestamp. That is the only field that can be
// joined against the recording segments, which are themselves timestamped by the
// filesystem. Without it you have detections you cannot play back -- you know a
// person was seen on camera 1 at frame 8,412 and no way to find the video.
// pts_ns is kept too, but it is a stream clock: it restarts at zero on every
// reconnect, so it can order frames within a run and nothing more.
//
// `new_ids` lists the track ids seen for the FIRST time on that camera. That is
// what turns "1,400 lines of boxes" into "a person arrived at 14:03:07" without
// a second file format -- the search tool reads one file either way.
//
// The element sits AFTER nvtracker, which is what makes `id` a persistent object
// id rather than a per-frame index, and it is a PASS-THROUGH, so the video
// carries on to the OSD and the wall. It reads metadata only and never touches a
// pixel.
//
// CONFIGURATION (env vars -- the plugin has no config-file property)
//   VMS_ROOT         ./store   events go to <VMS_ROOT>/events/cam<N>.jsonl
//   VMS_MIN_CONF     0.3       drop objects below this DETECTOR confidence.
//                              An object carrying a CLASSIFIER RESULT is kept
//                              whatever its confidence: nvinfer never assigns
//                              the classification placeholder's, so it reads
//                              0.0f and no gate above zero could admit it.
//                              A classifier's own gate is `classifier-threshold`
//                              in its nvinfer config, which is where nvinfer
//                              applies it -- and it has to be, because top-1 on
//                              a 1000-class model is routinely below 0.3 (this
//                              file's own re-run: 0.16 for "sports car"). A VMS
//                              gate re-applied to a softmax would silently
//                              re-create the bug it is documented under.
//   VMS_INTERVAL     5         log every Nth frame (30 fps / 5 = 6 lines/s/cam)
//   VMS_SKIP_EMPTY   1         1 = never log a frame with no objects AND no
//                              published shapes
//   VMS_STATUS       2.0       seconds between terminal status lines (0 = silent)
//   VMS_MAX_SHAPES   64        cap on forwarded render shapes per line
//                              (0 = do not forward shapes at all)

#include <sys/stat.h>
#include <sys/time.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "nvcustomprocess.h"
#include "nvdsmeta.h"

#include "event_sink.h"
#include "shape_json.h"

// render_meta.h is STRICTLY OPTIONAL. It only matters to a pipeline that has a
// custom render stage, so compile against it when it is there and behave
// exactly as before when it is not. No build-order coupling, no second
// mandatory file in parsers/, and a half-applied rollout cannot break a build
// -- it fails SOFT, as an absent shapes forward rather than a compile error.
// (frame_json.h guards its copy the same way, for the same reason.)
#if defined(__has_include)
#  if __has_include("render_meta.h")
#    include "render_meta.h"
#    define VMS_DETECT_WITH_SHAPES 1
#  endif
#  if __has_include("analytics_meta.h")
#    include "analytics_meta.h"
#    define VMS_DETECT_WITH_ANALYTICS 1
#  endif
#endif

namespace {

std::mutex g_lock;
bool g_init = false;

std::string g_root = "./store";
double g_min_conf = 0.3;
long g_interval = 5;
bool g_skip_empty = true;
double g_status_every = 2.0;
long g_max_shapes = 64;

// One open file per camera. Opened lazily: a source that never produces a frame
// should not leave an empty log behind suggesting it did.
event_sink::Server g_sink;
event_sink::Tiles g_tiles;
std::map<unsigned, FILE*> g_files;
// Track ids already seen, per camera -- the basis of `new_ids`.
std::map<unsigned, std::set<uint64_t> > g_seen;

uint64_t g_frames = 0, g_lines = 0, g_objects = 0;
double g_t0 = 0.0, g_t_status = 0.0;
uint64_t g_lines_at_status = 0;

double now_epoch() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double) tv.tv_sec + (double) tv.tv_usec * 1e-6;
}

const char* env_or(const char* k, const char* d) {
    const char* v = getenv(k);
    return (v && *v) ? v : d;
}

void lazy_init() {
    if (g_init) return;
    g_init = true;
    g_root = env_or("VMS_ROOT", "./store");
    g_min_conf = atof(env_or("VMS_MIN_CONF", "0.3"));
    g_interval = atol(env_or("VMS_INTERVAL", "5"));
    if (g_interval < 1) g_interval = 1;
    g_skip_empty = atoi(env_or("VMS_SKIP_EMPTY", "1")) != 0;
    g_status_every = atof(env_or("VMS_STATUS", "2.0"));
    // A liveness budget, not a formatting one -- see ShapeViews::forward below.
    // Read once here rather than per frame; 0 disables shape forwarding.
    g_max_shapes = atol(env_or("VMS_MAX_SHAPES", "64"));
    if (g_max_shapes < 0) g_max_shapes = 0;
    g_t0 = g_t_status = now_epoch();
    g_tiles.load();
    g_sink.start(atoi(env_or("VMS_META_PORT", "7002")));
    g_t0 = g_t_status = now_epoch();
    // mkdir rather than assume: run.sh makes this, but the library is also usable
    // from a hand-written pipeline that does not.
    std::string dir = g_root + "/events";
    mkdir(g_root.c_str(), 0755);
    mkdir(dir.c_str(), 0755);
    fprintf(stderr, "[vms-events] -> %s/events/cam<N>.jsonl  "
                    "(min_conf=%.2f interval=%ld skip_empty=%d)\n",
            g_root.c_str(), g_min_conf, g_interval, (int) g_skip_empty);
}

FILE* file_for(unsigned source_id) {
    std::map<unsigned, FILE*>::iterator it = g_files.find(source_id);
    if (it != g_files.end()) return it->second;
    char path[1024];
    snprintf(path, sizeof(path), "%s/events/cam%u.jsonl", g_root.c_str(), source_id);
    // APPEND, never truncate. A VMS restart must not erase the day's events --
    // retention is prune.py's job and it is the only thing allowed to delete.
    FILE* f = fopen(path, "a");
    if (!f) {
        fprintf(stderr, "[vms-events] cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    // Line buffered: a reader tailing the log sees events as they happen, and a
    // hard kill loses at most the line in flight rather than a 4 KB block.
    setvbuf(f, NULL, _IOLBF, 0);
    g_files[source_id] = f;
    return f;
}

// ── what a classifier attached to an object ─────────────────────────────────
//
// WHY THE ANSWER GOES ONTO THE BOX AND NOT INTO `fields`. shape_json's
// vocabulary is CLOSED at box/polygon/polyline/skeleton, and that closure is
// the property worth keeping: a console that draws four primitives renders a
// pipeline it has never seen. A classification result is not a new geometry --
// it is a NAME for a region that already exists -- so it belongs where every
// other name belongs, in that region's own `label`/`score`, with any further
// results in `text`. `fields` was the obvious alternative and is wrong here:
// field keys are per-LINE, not per-shape, so a detector with a secondary
// classifier on twelve objects would write the same key twelve times and every
// JSON reader keeps only the last. No new `kind` reaches the console, no key
// changes shape, and the console needs no change at all.
enum {
    // Matches frame_json.h's kMaxAttributes, so the three metadata sinks agree
    // on how many results one object can carry.
    kMaxAttrs    = 32,
    kMaxAttrWalk = 4096,   // nodes visited in one object's metadata lists
    kMaxAttrText = 200     // bytes of `text` summary per object (see attr_text)
};

struct Attr {
    const NvDsClassifierMeta* cm;
    const NvDsLabelInfo*      li;
};

// A BOUNDED, UTF-8-VALIDATING copy. Not a strcpy, for two independent reasons.
//
//   * NvDsLabelInfo::result_label is a FIXED 128-byte array filled with
//     strncpy and is NOT guaranteed to be NUL-terminated, so the read must be
//     bounded by the array rather than by a terminator that may not be there.
//   * shape_json::escape() passes every byte >= 0x20 through unexamined. A
//     model shipped with a Latin-1 labels file would therefore put raw
//     0x80..0xFF into the .jsonl and onto the socket, and the far end fails on
//     the WHOLE LINE -- json.loads() raises UnicodeDecodeError, and a console
//     drops a line it cannot parse. One bad label must cost one field.
//
// Malformed sequences become U+FFFD (RFC 3629: shortest form, no surrogates).
// For pure-ASCII labels -- everything an ImageNet- or COCO-style labels file
// actually contains -- this is a byte-for-byte copy.
void utf8_copy(std::string& out, const char* s, size_t max) {
    out.clear();
    if (!s) return;
    const unsigned char* p = (const unsigned char*) s;
    size_t n = 0;
    while (n < max && p[n]) ++n;
    size_t i = 0;
    while (i < n) {
        const unsigned char c = p[i];
        if (c < 0x80u) { out += (char) c; ++i; continue; }
        size_t need = 0;
        unsigned char lo = 0x80u, hi = 0xBFu;
        if      (c >= 0xC2u && c <= 0xDFu) need = 1;
        else if (c == 0xE0u)               { need = 2; lo = 0xA0u; }
        else if (c >= 0xE1u && c <= 0xECu) need = 2;
        else if (c == 0xEDu)               { need = 2; hi = 0x9Fu; }
        else if (c >= 0xEEu && c <= 0xEFu) need = 2;
        else if (c == 0xF0u)               { need = 3; lo = 0x90u; }
        else if (c >= 0xF1u && c <= 0xF3u) need = 3;
        else if (c == 0xF4u)               { need = 3; hi = 0x8Fu; }
        bool ok = (need != 0) && (i + need < n);
        for (size_t k = 0; ok && k < need; ++k) {
            const unsigned char cc  = p[i + 1 + k];
            const unsigned char klo = (k == 0) ? lo : 0x80u;
            const unsigned char khi = (k == 0) ? hi : 0xBFu;
            if (cc < klo || cc > khi) ok = false;
        }
        if (ok) { out.append((const char*) p + i, need + 1); i += need + 1; }
        else    { out += "\xEF\xBF\xBD"; ++i; }
    }
}

// pResult_label carries the string when it did not fit the fixed array;
// NVIDIA's own consumers prefer it, so we do too.
void attr_label(std::string& out, const NvDsLabelInfo* li) {
    if (li->pResult_label) utf8_copy(out, li->pResult_label, 256);
    else                   utf8_copy(out, li->result_label, MAX_LABEL_SIZE);
}

// shape_json::tail() formats `score` with "%.4f" and emits it only when
// score > 0.0. Every comparison with a NaN is false, so a NaN probability is
// already dropped by that gate; an INFINITY is not, and "%.4f" of one is `inf`
// -- a token no JSON parser accepts, which would cost the whole line rather
// than one field. Both are refused here instead.
double attr_score(const NvDsLabelInfo* li) {
    const double p = (double) li->result_prob;
    return (p > 0.0 && p < 1.0e9) ? p : 0.0;
}

// Every classifier result on one object, in the order nvinfer produced them.
//
// Both lists are built with g_list_prepend, so walking them forwards yields
// REVERSE model order. Results are gathered into a caller-provided fixed buffer
// -- no heap traffic on the per-object path -- and then reversed, which
// restores the attribute / output-layer order the model declared. That matters
// for a multi-output classifier (vehicle type, then colour), whose layers would
// otherwise reach the console inverted. The walk is capped so a corrupted or
// cyclic list costs a truncated result rather than the streaming thread.
size_t collect_attrs(const NvDsObjectMeta* ob, Attr* out, size_t cap) {
    size_t n = 0, walked = 0;
    if (!ob) return 0;
    for (NvDsMetaList* cl = ob->classifier_meta_list;
         cl && n < cap && walked < (size_t) kMaxAttrWalk; cl = cl->next) {
        ++walked;
        const NvDsClassifierMeta* cm = (const NvDsClassifierMeta*) cl->data;
        if (!cm) continue;
        for (NvDsMetaList* ll = cm->label_info_list;
             ll && n < cap && walked < (size_t) kMaxAttrWalk; ll = ll->next) {
            ++walked;
            const NvDsLabelInfo* li = (const NvDsLabelInfo*) ll->data;
            if (!li) continue;
            out[n].cm = cm;
            out[n].li = li;
            ++n;
        }
    }
    for (size_t i = 0, j = n; i + 1 < j; ++i) {
        --j;
        const Attr t = out[i]; out[i] = out[j]; out[j] = t;
    }
    return n;
}

// The results that did NOT become the box's own label, as one readable string:
//
//     vehicletype=sedan 0.87 | colour=silver 0.62
//
// classifier_type is unset unless a config names one, which is the usual case,
// and then only the label appears. A result whose label is empty is shown as
// its class id (#207) rather than dropped -- an id is still an answer, and a
// silently missing attribute is the failure this whole change exists to end.
//
// BOUNDED: the length test is at the top of the iteration, so the worst case is
// kMaxAttrText plus one element (a 64-byte type, a 128-byte label, a score) --
// well inside shape_json::escape(), which grows a std::string and cannot
// truncate, unlike appendf's 512-byte buffer.
void attr_text(std::string& out, const Attr* a, size_t n) {
    std::string type, label;
    char b[32];
    for (size_t i = 0; i < n && out.size() < (size_t) kMaxAttrText; ++i) {
        if (!out.empty()) out += " | ";
        if (a[i].cm->classifier_type && *a[i].cm->classifier_type)
            utf8_copy(type, a[i].cm->classifier_type, 64);
        else
            type.clear();
        if (!type.empty()) { out += type; out += '='; }
        attr_label(label, a[i].li);
        if (!label.empty()) {
            out += label;
        } else {
            snprintf(b, sizeof(b), "#%u", (unsigned) a[i].li->result_class_id);
            out += b;
        }
        const double p = attr_score(a[i].li);
        if (p > 0.0) { snprintf(b, sizeof(b), " %.2f", p); out += b; }
    }
}

// Native analytics keeps zero-object frames (zero occupancy and cumulative
// counters are still useful) and tracked objects with detector confidence -0.1.
inline bool has_analytics_frame(const NvDsFrameMeta* fm) {
#ifdef VMS_DETECT_WITH_ANALYTICS
    return analytics_meta::has_frame(fm);
#else
    return false;
#endif
}
inline bool has_analytics_object(const NvDsObjectMeta* ob) {
#ifdef VMS_DETECT_WITH_ANALYTICS
    return analytics_meta::has_object(ob);
#else
    return false;
#endif
}

// ── the shapes a render stage published ─────────────────────────────────────
//
// TWO DEFINITIONS OF ONE TYPE, so VmsDetect below reads identically whether or
// not render_meta.h is present. Without it every method is trivially empty and
// `empty()` is a compile-time-constant `true`, which is what makes the `due`
// gate below reduce to exactly the expression it has always been -- the
// alternative, #ifdefs sprinkled through the per-frame path, is where a
// "byte-identical when absent" claim quietly stops being true.
#ifdef VMS_DETECT_WITH_SHAPES
struct ShapeViews {
    std::vector<render_meta::View> v;

    // Every valid payload on this frame, in publish order. A frame with none --
    // which is EVERY frame of every pipeline that has no render stage -- leaves
    // `v` empty and nothing below does anything at all.
    void collect(NvDsFrameMeta* fm) {
        if (g_max_shapes > 0) render_meta::views_on_frame(fm, v);
    }
    bool empty() const { return v.empty(); }

    // EXACTLY the transform the box loop applies, and for the same reason: a
    // shift into this source's tile, and NOTHING ELSE.
    //
    // render_meta::View offers scale_x/scale_y for a consumer whose pixel space
    // differs from the publisher's, and it is deliberately NOT used here. The
    // publisher declares in->W/in->H, which is the nvstreammux output size --
    // the very space NvDsObjectMeta::rect_params live in -- so shapes and boxes
    // are already in one space and any scaling would move one and not the
    // other. It would also be scaling by a number nobody set: run.vms.sh
    // exports VMS_ROOT/MIN_CONF/INTERVAL/SKIP_EMPTY/STATUS/META_PORT/
    // MJPEG_PORT/JPEG_QUALITY and NOT the VMS_TILE_* group, so g_tiles holds
    // its 1280x720 defaults on every generated pipeline whatever the mux size.
    // Scaling to that would silently shrink a 1920x1080 pipeline's skeletons
    // while leaving its boxes alone. Shapes and boxes must be wrong together or
    // right together; that is the property worth having.
    //
    // THE CAP IS A LIVENESS GUARD. event_sink drops a console that cannot drain
    // and counts a partial send as a failure, so one runaway decoder must not
    // be able to knock every console off the wall. It is spent across payloads
    // (`cap - n`), not per payload, because the budget is the LINE.
    size_t forward(shape_json::Line& line, double ox, double oy) const {
        const size_t cap = (size_t) g_max_shapes;
        size_t n = 0;
        for (size_t i = 0; i < v.size() && n < cap; ++i)
            n += render_meta::to_shape_line(v[i], line, ox, oy, 1.0, 1.0,
                                            cap - n);
        return n;
    }
};
#else
struct ShapeViews {
    void collect(NvDsFrameMeta*) {}
    bool empty() const { return true; }
    size_t forward(shape_json::Line&, double, double) const { return 0; }
};
#endif

}  // namespace

extern "C" bool VmsDetect(NvDsCustomPostProcessFrameInCpu* in)
{
    std::lock_guard<std::mutex> guard(g_lock);
    lazy_init();

    NvDsFrameMeta* fm = (NvDsFrameMeta*) in->frame_meta;
    // Returning false ABORTS the whole pipeline. A VMS must keep recording and
    // keep showing the wall even if the event log cannot be written -- losing the
    // video because a disk filled up is a worse outcome than losing the events.
    // So every failure path here lets the frame through.
    if (!fm) return true;

    std::vector<NvDsObjectMeta*> objs;
    objs.reserve(in->num_obj_metas);
    for (size_t i = 0; i < in->num_obj_metas; ++i) {
        NvDsObjectMeta* ob = (NvDsObjectMeta*) in->obj_metas[i];
        // OR, not AND, and the OR is not optional. nvinfer NEVER ASSIGNS the
        // whole-frame classification placeholder's confidence -- it reads
        // 0.0f out of a zeroed pool slot -- so ANY gate above zero drops the
        // only object a classification pipeline produces, and VMS_SKIP_EMPTY
        // then drops the frame with it. The generated default is 0.3.
        //
        // A detector's objects have no classifier_meta_list, so on every
        // detection pipeline this test evaluates exactly as it always did.
        // (Measured: the production detection line is byte-identical.)
        //
        // mask_params is deliberately NOT part of this test, even though
        // frame_json::has_payload() -- the same idea, for the JSON and MQTT
        // sinks -- also tests it. Those sinks have a mask{} key to put an
        // instance mask in; this one has no mask renderer, so admitting a
        // sub-threshold instance-seg object would weaken the operator's
        // confidence filter and buy an unlabelled box. Add it here in the same
        // breath as a producer that draws the outline, and not before.
        if (ob && (ob->confidence >= g_min_conf || ob->classifier_meta_list
                   || has_analytics_object(ob)))
            objs.push_back(ob);
    }

    ++g_frames;
    const bool on_interval = (fm->frame_num % (unsigned) g_interval) == 0;

    // Gathered BEFORE `due`, because published shapes are themselves a reason
    // to log a frame. A pose-, lane- or depth-only pipeline has no
    // NvDsObjectMeta at all, so VMS_SKIP_EMPTY (on by default) drops every one
    // of its frames -- which is why such a pipeline's event log today is EMPTY
    // rather than wrong, and why relaxing this gate is not optional. Only ever
    // consulted on an on-interval frame, so it costs nothing on the frames that
    // were going to be skipped anyway.
    ShapeViews shapes;
    if (on_interval) shapes.collect(fm);

    // With no shapes -- every frame of every pipeline without a render stage,
    // and every frame at all in a build with no render_meta.h -- the third
    // conjunct is a constant true and this is exactly the expression it has
    // always been.
    const bool due = on_interval
                     && !(g_skip_empty && objs.empty() && shapes.empty()
                          && !has_analytics_frame(fm));

    if (due) {
        // First-seen ids for THIS camera. Computed before the line is built so
        // the line can carry them.
        std::set<uint64_t>& seen = g_seen[fm->source_id];
        std::vector<uint64_t> fresh;
        for (size_t i = 0; i < objs.size(); ++i) {
            const uint64_t id = (uint64_t) objs[i]->object_id;
            if (objs[i]->object_id == UNTRACKED_OBJECT_ID) continue;
            if (seen.insert(id).second) fresh.push_back(id);
        }
        // A long-running camera accumulates ids without bound. Ids are monotonic
        // per source, so forgetting the oldest half costs nothing real: an id
        // that old will never be re-issued and so can never be reported "new"
        // again by mistake.
        if (seen.size() > 200000) {
            std::set<uint64_t>::iterator mid = seen.begin();
            std::advance(mid, seen.size() / 2);
            seen.erase(seen.begin(), mid);
        }

        FILE* f = file_for(fm->source_id);
        if (f) {
            shape_json::Line line(now_epoch(), fm->source_id, fm->frame_num,
                                  g_tiles.space_w, g_tiles.space_h);
            // Shift into this source's tile. The tiler composed N streams onto
            // one canvas, so geometry in per-source coordinates describes
            // nothing the console can draw over the composed frame.
            const double ox = g_tiles.x_of(fm->source_id);
            const double oy = g_tiles.y_of(fm->source_id);
            for (size_t i = 0; i < objs.size(); ++i) {
                NvDsObjectMeta* ob = objs[i];
                const NvOSD_RectParams& r = ob->rect_params;
                const int64_t id = (ob->object_id == UNTRACKED_OBJECT_ID)
                                       ? -1 : (int64_t) ob->object_id;
                // Whatever a classifier attached, if anything. A detector's
                // object has no classifier_meta_list at all: `na` is 0, the
                // three values below are exactly what they have always been,
                // `text` stays empty and the call is the call it was. That is
                // what keeps a detection line byte-identical, and it is
                // structural rather than lucky.
                Attr attrs[kMaxAttrs];
                const size_t na = collect_attrs(ob, attrs, (size_t) kMaxAttrs);
                const char* label = ob->obj_label;
                double score = ob->confidence;
                std::string adopted, text;
                size_t first_text = 0;
                if (na) {
                    // The box adopts the first result ONLY where the object has
                    // no name of its own -- which is precisely the whole-frame
                    // classification placeholder, whose obj_label is "". A
                    // detector+SGIE object keeps its own "car" and its own
                    // detector score and sends its attributes to `text`: the
                    // console must never be told an SGIE's confidence is the
                    // detector's.
                    //
                    // Label and score are adopted TOGETHER. A score has to
                    // describe the thing named beside it, so a box labelled
                    // from a classifier result must not keep a number that came
                    // from somewhere else.
                    if (!label || !*label) {
                        attr_label(adopted, attrs[0].li);
                        if (!adopted.empty()) {
                            label = adopted.c_str();
                            score = attr_score(attrs[0].li);
                            first_text = 1;
                        }
                    }
                    attr_text(text, attrs + first_text, na - first_text);
                }
                line.box(r.left + ox, r.top + oy, r.width, r.height,
                         label, id, score,
                         text.empty() ? NULL : text.c_str());
#ifdef VMS_DETECT_WITH_ANALYTICS
                analytics_meta::forward_object(line, ob, i);
#endif
            }
            // Then whatever a render stage decoded, through the SAME writers
            // and into the SAME shapes array -- detector boxes first because
            // they are the frame's primary subject and a console renders in
            // order. Appends nothing when nothing was published, which is what
            // keeps a detection line byte-identical. Any scalars the renderer
            // published land in `fields` ahead of `objects` for the same
            // reason: `objects` is this file's own summary and reads last.
            shapes.forward(line, ox, oy);
#ifdef VMS_DETECT_WITH_ANALYTICS
            analytics_meta::forward_frame(line, fm);
#endif
            line.field_i("objects", (long long) objs.size());
            if (!fresh.empty())
                line.field_i("new", (long long) fresh.size());

            std::string text = line.finish();
            // `new_ids` is spliced in rather than being a shape field: it is a
            // property of the FRAME (which ids appeared here for the first time),
            // which is what the console's "first sightings only" filter needs.
            if (!fresh.empty()) {
                std::string ids = ", \"new_ids\": [";
                for (size_t i = 0; i < fresh.size(); ++i)
                    shape_json::appendf(ids, "%s%" PRIu64, i ? ", " : "", fresh[i]);
                ids += "]";
                text.insert(text.size() - 2, ids);   // before the closing "}\n"
            }
            fwrite(text.data(), 1, text.size(), f);
            g_sink.poll_accept();
            g_sink.broadcast(text);
            ++g_lines;
            g_objects += objs.size();
        }
    }

    if (g_status_every > 0.0) {
        const double now = now_epoch();
        if (now - g_t_status >= g_status_every) {
            fprintf(stderr, "[vms-events] %6" PRIu64 " frames  %6" PRIu64 " events  "
                            "%7" PRIu64 " objects  %5.1f ev/s  %zu camera(s)\n",
                    g_frames, g_lines, g_objects,
                    (double) (g_lines - g_lines_at_status) / (now - g_t_status),
                    g_files.size());
            g_t_status = now;
            g_lines_at_status = g_lines;
        }
    }
    return true;
}
