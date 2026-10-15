// frame_json.h -- the payload shape, in one place.
//
// SHARED by both metadata sinks PipeGen can emit: the MQTT publisher
// (agents/mqtt_agent.py) and the JSON file writer (agents/json_agent.py). Both
// write this same file into parsers/ and both compile against it, so a
// subscriber on a bus and a reader of a .json on disk see byte-identical
// objects. One constant rather than two copies -- the reference pipelines keep
// a copy each and their comments already warn about the drift that invites.
//
// WHAT A FRAME RECORD CAN CARRY. DeepStream does not put every model family in
// NvDsObjectMeta. A detector fills it; everything else hangs its answer
// somewhere else, which is why reading only the object meta makes a
// classification pipeline emit one empty-looking object per frame and a pose
// pipeline emit nothing at all. So a record grows four OPTIONAL keys, each read
// from the one place DeepStream actually puts that family:
//
//   objects[].attributes[]  <- obj->classifier_meta_list   (classification)
//   objects[].mask{}        <- obj->mask_params            (instance seg)
//   segmentation[]          <- NVDSINFER_SEGMENTATION_META (semantic seg)
//   analytics[]             <- native NvDsAnalyticsFrameMeta / ObjInfo
//                              (frame counts and per-object statuses, by instance)
//   shapes[] / fields{}     <- render_meta.h               (pose, lanes, depth:
//                              whatever a custom render .so decoded)
//
// EVERY ONE OF THOSE KEYS IS OMITTED ENTIRELY WHEN ITS METADATA IS ABSENT --
// not emitted as null, not emitted as []. On a detection pipeline every guard
// below fails on a NULL pointer before a single byte is appended, so a
// detector's objects[] is byte-for-byte what it was before any of this existed.
// That is a hard requirement, not a nicety: the detection JSON path is in
// production use and its output is diffed byte for byte.
//
// WHY EACH FAMILY IS READ THE WAY IT IS
//   attributes[]   is gated on classifier_meta_list != NULL, NEVER on
//                  confidence. nvinfer's full-frame classification placeholder
//                  has class_id = -1, an empty obj_label and confidence exactly
//                  0.0f -- the object pool zeroes released slots and
//                  attach_metadata_classifier never assigns it -- so a
//                  confidence gate above zero drops every classification result.
//                  See has_payload() at the bottom of this file.
//   mask{}         the grid is in the OBJECT'S OWN box space, not frame pixels;
//                  nvdsosd stretches it onto that object's rect_params. So
//                  `coverage` -- the fraction of cells over the model's
//                  threshold -- is the honest scalar, because it needs no
//                  coordinate system at all, and the opt-in outline is mapped
//                  back through rect_params into FRAME pixels before it is
//                  emitted. mask_params.size is in BYTES, not elements. A grid
//                  too large to scan cell by cell within the streaming thread's
//                  budget is SAMPLED, and says so -- see kMaxMaskScan.
//   segmentation[] is derived from class_map ONLY. The header documents
//                  class_probabilities_map as (c*w*h + y*w + x), but that is
//                  the NCHW layout; with segmentation-output-order=1 the
//                  producer writes NHWC and the struct carries no field saying
//                  which, so arithmetic on it is silently wrong for half of all
//                  models. class_map is already argmaxed and layout-independent
//                  -- and it legitimately contains -1 for "nothing beat the
//                  threshold", which is bucketed separately and NEVER used to
//                  index a labels array.
//
// WHAT IS DELIBERATELY NOT HERE: #include "gstnvdsinfer.h". That header has no
// include guard and references GstBuffer without including <gst/gst.h> itself,
// while the Makefile rules for these sinks are glib-only -- pulling it in
// breaks the build. The one type needed from it, NvDsInferSegmentationMeta, is
// mirrored below and static_asserted.
//
// OUTPUT IS BOUNDED. appendf() below truncates at 511 bytes PER CALL and does
// so silently, which would produce invalid JSON. Every list here is therefore
// emitted ONE ELEMENT PER CALL, and every list has a hard element cap.

#ifndef FRAME_JSON_H_
#define FRAME_JSON_H_

#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsmeta.h"

// shapes[] is STRICTLY OPTIONAL. render_meta.h only matters to a pipeline that
// has a custom render stage, so compile against it when it is there and behave
// exactly as before when it is not. No build-order coupling, no second
// mandatory file in parsers/, and a half-applied rollout cannot break a build.
#if defined(__has_include)
#  if __has_include("render_meta.h")
#    include "render_meta.h"
#    define FRAME_JSON_WITH_SHAPES 1
#  endif
#  if __has_include("analytics_meta.h")
#    include "analytics_meta.h"
#    define FRAME_JSON_WITH_ANALYTICS 1
#  endif
#endif

namespace frame_json {

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

// -- caps --------------------------------------------------------------------
//
// Not tuning knobs: the reason a malformed frame costs a truncated key instead
// of an unbounded document or a stalled streaming thread. EVERY loop in this
// file that reads producer-supplied metadata is bounded by one of these before
// it dereferences anything.
enum {
    kMaxAttributes = 32,        // classifier results emitted per object
    kMaxListWalk   = 4096,      // nodes visited in any one metadata list
    kMaxOutlineRow = 32,        // sampled mask rows -> at most 64 outline points
    kMaxMaskCells  = 4194304,   // 2048x2048; anything bigger is a decode bug
    kMaxMaskScan   = 65536,     // mask cells read exactly before sampling starts
    kMaxSegMetas   = 8,         // segmentation metas read from one frame
    kMaxSegClasses = 4096,      // histogram width
    kMaxSegCells   = 33554432,  // 8192x4096; an 8K class map still fits
    kMaxSegEmit    = 64,        // classes listed per segmentation meta
    kSegStride     = 4          // sample every 4th pixel on BOTH axes
};

// Module-local linkage for anything in this header that owns a function-local
// static. GCC gives a plain inline function's local statics STB_GNU_UNIQUE
// binding, which the dynamic linker unifies PROCESS-WIDE: with libmeta_json.so
// and libmqtt_publish.so both loaded, the variable would physically live in
// whichever object the loader relocated first, and every read from the other
// one would reach across a module boundary it cannot see. nvcustomprocess
// dlclose()s each custom .so from GstBaseTransform::stop on PAUSED->READY, so
// that is a use-after-unmap waiting for the right teardown order. Hidden
// visibility demotes the symbol to module-local BSS -- `nm` reports `b` instead
// of `u` -- and each module keeps its own copy, which is correct here because
// the value is a pure function of the process's environment. render_meta.h
// defends its own statics the same way, for the same reason.
#if defined(__GNUC__)
#  define FRAME_JSON_LOCAL __attribute__((visibility("hidden")))
#else
#  define FRAME_JSON_LOCAL
#endif

// -- a stricter string writer, for the NEW keys only -------------------------
//
// escape() above is what every EXISTING key goes through, and it is left
// exactly as it was: touching it could move a byte of the detection payload.
// The new keys use this instead, because their inputs are worse.
//
//   * It never reads past `max` bytes. NvDsLabelInfo::result_label is a fixed
//     128-byte array filled with strncpy, so it is not guaranteed to be
//     terminated.
//   * It validates UTF-8 (RFC 3629: shortest form, no surrogates) and replaces
//     every malformed byte with U+FFFD. A model shipped with a Latin-1 labels
//     file would otherwise put raw 0x80..0xFF bytes into the document, and
//     json.loads() on the far end raises UnicodeDecodeError -- one bad label
//     costs the whole run's payload, not one field.
//
// For pure-ASCII input it is byte-for-byte identical to escape(), so the two
// can never disagree about a label a COCO-style model actually produces.
inline void escape_n(std::string& out, const char* s, size_t max) {
    out += '"';
    const unsigned char* p = (const unsigned char*) s;
    size_t n = 0;
    if (p) { while (n < max && p[n]) ++n; }
    size_t i = 0;
    while (i < n) {
        const unsigned char c = p[i];
        if (c < 0x80u) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
                    else out += (char) c;
            }
            ++i;
            continue;
        }
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
        bool okseq = (need != 0) && (i + need < n);
        for (size_t k = 0; okseq && k < need; ++k) {
            const unsigned char cc  = p[i + 1 + k];
            const unsigned char klo = (k == 0) ? lo : 0x80u;
            const unsigned char khi = (k == 0) ? hi : 0xBFu;
            if (cc < klo || cc > khi) okseq = false;
        }
        if (okseq) { out.append((const char*) p + i, need + 1); i += need + 1; }
        else       { out += "\\ufffd"; ++i; }
    }
    out += '"';
}

// "%.4f" of a NaN is `nan` and of an infinity is `inf` -- both are tokens no
// JSON parser accepts, and one of them in a line-delimited log costs the whole
// run's output. Every float on a NEW key goes through here.
inline double safe_num(double v) { return std::isfinite(v) ? v : 0.0; }

// -- objects[].attributes[] : classifier_meta_list ---------------------------
//
// nvinfer in PGIE classifier mode fabricates one whole-frame NvDsObjectMeta per
// frame with class_id = -1, an empty obj_label and an unassigned confidence.
// The real result is in classifier_meta_list -> NvDsClassifierMeta ->
// label_info_list -> NvDsLabelInfo, and nothing read it.
//
// Both lists are built with g_list_prepend, so walking them forwards yields
// REVERSE model order. Results are gathered into a fixed buffer -- no heap
// traffic on the per-object hot path -- and replayed backwards, which restores
// the order nvinfer produced (attribute index / output-layer order). That
// matters for a multi-output classifier, whose layers would otherwise appear
// inverted.
//
// THE BUFFER IS A RING, AND THAT IS THE WHOLE POINT. Stopping the forward walk
// once kMaxAttributes results are in hand would keep the WRONG ones: the walk
// meets the LAST-inserted label first, so an early stop retains the highest
// attribute indices and throws away index 0. A top-k classifier's parser
// pushes its results in DESCENDING confidence
// (gstnvinfer_meta_utils.cpp:278-291 loops i = 0..num_attrs forward and calls
// nvds_add_label_info_meta_to_classifier, which prepends), so an early stop
// discards the model's BEST answers and publishes only its worst -- silently,
// and in a way no consumer can detect. Retaining the last kMaxAttributes seen
// instead keeps insertion indices 0..kMaxAttributes-1, i.e. the FIRST results
// the model produced, which is what "capped at 32" has to mean to be worth
// anything. The cap costs at most kMaxListWalk pointer-chases and no
// allocation; when a truncation happens at all, attributes_omitted says so.
inline void attributes(std::string& out, NvDsObjectMeta* ob) {
    if (!ob || !ob->classifier_meta_list) return;

    const NvDsClassifierMeta* owner[kMaxAttributes];
    const NvDsLabelInfo*      info[kMaxAttributes];
    size_t seen = 0, walked = 0;   // `seen` counts ALL results, past the cap

    for (NvDsMetaList* cl = ob->classifier_meta_list;
         cl && walked < (size_t) kMaxListWalk;
         cl = cl->next) {
        ++walked;
        const NvDsClassifierMeta* cm = (const NvDsClassifierMeta*) cl->data;
        if (!cm) continue;
        for (NvDsMetaList* ll = cm->label_info_list;
             ll && walked < (size_t) kMaxListWalk;
             ll = ll->next) {
            ++walked;
            const NvDsLabelInfo* li = (const NvDsLabelInfo*) ll->data;
            if (!li) continue;
            const size_t slot = seen % (size_t) kMaxAttributes;
            owner[slot] = cm;
            info[slot]  = li;
            ++seen;
        }
    }
    if (seen == 0) return;   // a classifier meta carrying no labels emits no key

    const size_t n = (seen < (size_t) kMaxAttributes) ? seen
                                                     : (size_t) kMaxAttributes;
    out += ", \"attributes\": [";
    // Walk position seen-1 is the FIRST label inserted, so counting down from
    // it replays the ring in model order. Below the cap the ring never wrapped
    // and this is the identical traversal a plain backwards replay performed.
    for (size_t k = 0; k < n; ++k) {
        if (k) out += ", ";
        const size_t slot = (seen - 1 - k) % (size_t) kMaxAttributes;
        const NvDsClassifierMeta* cm = owner[slot];
        const NvDsLabelInfo*      li = info[slot];
        // component_id tells several classifiers on one object apart when
        // classifier_type is unset, which it is unless a config names one.
        appendf(out, "{\"component_id\": %d", cm->unique_component_id);
        if (cm->classifier_type && *cm->classifier_type) {
            out += ", \"type\": ";
            escape_n(out, cm->classifier_type, 64);
        }
        out += ", \"label\": ";
        // pResult_label carries the string when it did not fit the fixed array;
        // NVIDIA's own consumers prefer it, so we do too.
        if (li->pResult_label) escape_n(out, li->pResult_label, 256);
        else                   escape_n(out, li->result_label, MAX_LABEL_SIZE);
        appendf(out, ", \"class_id\": %u, \"confidence\": %.4f}",
                (unsigned) li->result_class_id, safe_num((double) li->result_prob));
    }
    out += ']';
    // Truncation is announced, never silent -- the same contract segmentation[]
    // keeps with classes_omitted. A consumer must not have to guess whether a
    // 32-entry list is the whole answer.
    if (seen > n) appendf(out, ", \"attributes_omitted\": %" PRIu64,
                          (uint64_t) (seen - n));
}

// -- objects[].mask : mask_params --------------------------------------------
//
// The outline is up to 64 points per object per frame -- a different order of
// payload from a single coverage number -- so it is opt-in. Read once: a
// function-local static is initialised exactly once and thread-safely, which
// matters because the sinks run on the streaming thread of whichever branch
// they sit on, and getenv on every object of every frame is not free.
//
// JSON_MASK_OUTLINE is the documented name. META_MASK_OUTLINE is accepted as an
// alias because this is shared code: libmqtt_publish.so and libvms_detect.so
// also read it, and a JSON_-prefixed variable in those is a wart worth being
// able to retire without breaking anyone.
//
// FRAME_JSON_LOCAL is load-bearing, not decoration: this is the only
// function-local static in this header, and without it the symbol is
// STB_GNU_UNIQUE and shared across every .so in the process. See the macro's
// definition above.
FRAME_JSON_LOCAL inline bool mask_outline_on() {
    static const bool on = []() -> bool {
        const char* v = getenv("JSON_MASK_OUTLINE");
        if (!v || !*v) v = getenv("META_MASK_OUTLINE");
        if (!v || !*v) return false;
        return !(strcmp(v, "0") == 0 || strcmp(v, "false") == 0 ||
                 strcmp(v, "no") == 0 || strcmp(v, "off") == 0);
    }();
    return on;
}

inline void mask(std::string& out, NvDsObjectMeta* ob) {
    if (!ob) return;
    const NvOSD_MaskParams& m = ob->mask_params;
    if (!m.data || m.size == 0 || m.width == 0 || m.height == 0) return;

    const uint64_t cells = (uint64_t) m.width * (uint64_t) m.height;
    if (cells > (uint64_t) kMaxMaskCells) return;
    // `size` is in BYTES. A buffer that does not fully back the declared grid
    // is REFUSED, emitting no key at all -- not read short. Clamping to the
    // shorter of the two would be memory-safe but would publish a lie: a 16x16
    // mask whose "coverage" was computed from one float. nvinfer's own
    // producers always set size = width*height*sizeof(float) exactly, so this
    // rejects only garbage.
    if ((uint64_t) m.size < cells * sizeof(float)) return;

    float thr = m.threshold;
    if (!(thr > 0.0f && thr < 1.0f)) thr = 0.5f;   // also catches NaN

    // THE COVERAGE SCAN IS BUDGETED. This runs on the streaming thread of
    // whichever branch the sink sits on, once per object per frame, so its cost
    // is the pipeline's. Stock NVIDIA masks are the network's mask head --
    // 28x28, 784 cells, under a microsecond -- but PipeGen's own instance-seg
    // parser template mandates a mask "sized w x h ALIGNED TO THE BOX" in
    // network-input pixels, which is two to four orders of magnitude more
    // cells: measured here, ten 320x320 masks cost 470 us per frame and one
    // mask at kMaxMaskCells cost 1.9 ms. So the scan takes the coarsest
    // power-of-two stride that brings it under kMaxMaskScan cells, which leaves
    // stride 1 -- an EXACT, cell-by-cell scan -- for anything at or below that,
    // i.e. for every mask a stock parser produces and every case this file's
    // own documentation shows. `cells` is already <= kMaxMaskCells, so the
    // stride is <= 8 and the loop below cannot fail to terminate.
    //
    // Sampling is announced, never silent: `stride` and `sampled` appear ONLY
    // when a sample actually happened, so a mask scanned exactly emits exactly
    // the bytes it always did, and a reader is never told an estimate is a
    // count. It IS an estimate -- but a good one: measured against a full scan
    // over five patterns (solid blob, one-pixel diagonal, period-2
    // checkerboard, half-fill, soft sinusoid) at six sizes, the WORST error at
    // any stride is 0.0008 of coverage (the blob); every other pattern lands
    // within 0.0001, including the one-pixel diagonal that is far thinner than
    // its own stride. A consumer that needs a true pixel count still has
    // `stride` to tell it this is not one.
    unsigned stride = 1;
    while (cells / ((uint64_t) stride * (uint64_t) stride) >
           (uint64_t) kMaxMaskScan)
        stride <<= 1;

    // THE SAMPLE POINTS ARE A DIAGONAL LATTICE, NOT A GRID. A fixed column
    // offset makes the sample a plain subgrid, which ALIASES: measured, a
    // period-2 checkerboard sampled on any even stride reported coverage
    // 0.0000 against a true 0.5000, because every sample landed on the same
    // parity. Advancing the column phase by one cell per sampled row costs one
    // modulo per row and nothing per cell, and takes that same checkerboard to
    // 0.4980. At stride 1 the phase is always 0, so the common path is
    // untouched -- literally the same cells in the same order as before.
    //
    // `sampled` is accumulated per ROW rather than counted per cell for the
    // same reason: the inner loop stays one compare per cell and nothing else.
    uint64_t on = 0, sampled = 0;
    if (stride == 1) {
        // The exact path keeps its own flat loop over contiguous memory,
        // BYTE-IDENTICAL to what this function has always done, because a
        // nested loop carrying a RUNTIME stride is one the compiler cannot
        // vectorise: measured, folding the two cases together cost 40% on
        // twenty 160x160 masks (252 us -> 354 us) for no benefit whatsoever to
        // the case that actually needed help. This is the path every stock
        // NVIDIA mask takes.
        sampled = cells;
        for (uint64_t i = 0; i < cells; ++i)
            if (m.data[i] > thr) ++on;             // NaN compares false, as wanted
    } else {
        for (unsigned y = 0; y < m.height; y += stride) {
            const float* row = m.data + (uint64_t) y * m.width;
            const unsigned phase = (y / stride) % stride;
            if (phase >= m.width) continue;        // grid narrower than its stride
            sampled += (uint64_t) ((m.width - phase + stride - 1u) / stride);
            for (unsigned x = phase; x < m.width; x += stride)
                if (row[x] > thr) ++on;
        }
        if (sampled == 0) return;                  // emits no key rather than 0/0
    }

    appendf(out, ", \"mask\": {\"width\": %u, \"height\": %u, \"threshold\": %.4f,"
                 " \"coverage\": %.4f",
            (unsigned) m.width, (unsigned) m.height, (double) thr,
            (double) on / (double) sampled);
    if (stride > 1)
        appendf(out, ", \"stride\": %u, \"sampled\": %" PRIu64, stride, sampled);

    if (!mask_outline_on() || on == 0) { out += '}'; return; }

    // A ROW-SAMPLED SILHOUETTE, not a contour trace: up to 32 evenly spaced
    // rows contribute their leftmost and rightmost above-threshold cell, walked
    // down the right side and back up the left. O(rows*width) instead of
    // O(width*height), no allocation, no state machine, and at most 64 points
    // BY CONSTRUCTION -- properties a marching-squares trace does not have, and
    // the reason this maps onto the single simple closed ring that the existing
    // "polygon" primitive can express.
    //
    // It is an envelope, not the mask: concavities and holes are not
    // represented. That is the honest cost of a bounded, allocation-free
    // outline. Emitted in FRAME PIXELS, the same space as bbox.
    const NvOSD_RectParams& r = ob->rect_params;
    const double sx = (double) r.width  / (double) m.width;
    const double sy = (double) r.height / (double) m.height;
    const unsigned rows = (m.height < (unsigned) kMaxOutlineRow)
                              ? m.height : (unsigned) kMaxOutlineRow;

    double rx[kMaxOutlineRow], ry[kMaxOutlineRow];
    double lx[kMaxOutlineRow], ly[kMaxOutlineRow];
    size_t nr = 0, nl = 0;
    for (unsigned k = 0; k < rows; ++k) {
        // Centred sampling: k=0 is not the top edge, so a 1-row-tall mask still
        // samples its middle rather than its boundary.
        const unsigned y = (unsigned) ((((uint64_t) k * 2u + 1u) * m.height) / (rows * 2u));
        if (y >= m.height) continue;
        const float* row = m.data + (uint64_t) y * m.width;
        unsigned first = m.width, last = 0;
        for (unsigned x = 0; x < m.width; ++x) {
            if (!(row[x] > thr)) continue;
            if (first == m.width) first = x;
            last = x;
        }
        if (first == m.width) continue;            // nothing on this row
        const double yy = (double) r.top + ((double) y + 0.5) * sy;
        rx[nr] = (double) r.left + ((double) last  + 0.5) * sx; ry[nr] = yy; ++nr;
        // A one-cell-wide row contributes ONE vertex, not the same point twice:
        // a duplicated vertex is legal JSON but makes a degenerate edge that
        // some polygon renderers fill incorrectly.
        if (last == first) continue;
        lx[nl] = (double) r.left + ((double) first + 0.5) * sx; ly[nl] = yy; ++nl;
    }
    if (nr + nl < 3) { out += '}'; return; }       // two points are a line

    out += ", \"outline\": [";
    for (size_t i = 0; i < nr; ++i)                // right edge, top to bottom
        appendf(out, "%s[%.1f, %.1f]", i ? ", " : "", safe_num(rx[i]), safe_num(ry[i]));
    for (size_t i = nl; i-- > 0; )                 // left edge, back up
        appendf(out, ", [%.1f, %.1f]", safe_num(lx[i]), safe_num(ly[i]));
    out += "]}";
}

// -- segmentation[] : NVDSINFER_SEGMENTATION_META ----------------------------
//
// LAYOUT MIRROR of NvDsInferSegmentationMeta (gstnvdsinfer.h, DeepStream 7.1).
// Mirrored rather than included -- see the note at the top of this file. The
// enum that identifies it, NVDSINFER_SEGMENTATION_META, is already in
// nvdsmeta.h, so this needs no new include and no new compiler flag. The struct
// is plain C and has been stable since DeepStream 5; the asserts catch a
// packing surprise at build time rather than trusting that at runtime.
struct SegMeta {
    guint   classes;
    guint   width;
    guint   height;
    gint*   class_map;                  // pixel (x,y) at index y*width + x
    gfloat* class_probabilities_map;    // layout is order-dependent: UNUSED
    void*   priv_data;
    gint    unique_id;
};

#if defined(__LP64__) || defined(_LP64)
static_assert(sizeof(guint) == 4 && sizeof(gint) == 4,
              "SegMeta mirror assumes 32-bit guint/gint");
static_assert(offsetof(SegMeta, class_map) == 16,
              "NvDsInferSegmentationMeta layout changed");
static_assert(offsetof(SegMeta, class_probabilities_map) == 24,
              "NvDsInferSegmentationMeta layout changed");
static_assert(offsetof(SegMeta, priv_data) == 32,
              "NvDsInferSegmentationMeta layout changed");
static_assert(offsetof(SegMeta, unique_id) == 40,
              "NvDsInferSegmentationMeta layout changed");
static_assert(sizeof(SegMeta) == 48,
              "NvDsInferSegmentationMeta layout changed");
#endif

// A full class map is width*height integers per frame and nobody wants that in
// a line-delimited log. What a consumer can act on is how much of the frame
// each class covers, so the map is sampled on a 4-PIXEL STRIDE in both axes
// (1/16 of the pixels) and reduced to fractions. A 1024x1024 map costs 65k
// reads instead of a million, and coverage to four decimals does not need more.
// `stride` and `sampled` are emitted so a reader knows this is an estimate: a
// structure thinner than the stride can be under- or over-counted.
inline void segmentation(std::string& out, NvDsFrameMeta* fm) {
    if (!fm || !fm->frame_user_meta_list) return;

    // PASS 1 -- GATHER AND VALIDATE, dereferencing no class map yet.
    //
    // nvds_add_user_meta_to_frame prepends, so a forward walk yields REVERSE
    // publish order. The metas are collected into a fixed array and replayed
    // backwards, exactly as attributes() above does for classifier_meta_list
    // and render_meta.h does for its views: two segmentation stages on one
    // pipeline must not swap places, and a consumer must not have to sort by
    // unique_id to recover which ran first.
    const SegMeta* got[kMaxSegMetas];
    size_t n = 0, walked = 0;
    for (NvDsMetaList* l = fm->frame_user_meta_list;
         l && n < (size_t) kMaxSegMetas && walked < (size_t) kMaxListWalk;
         l = l->next) {
        ++walked;
        const NvDsUserMeta* um = (const NvDsUserMeta*) l->data;
        if (!um || um->base_meta.meta_type != (NvDsMetaType) NVDSINFER_SEGMENTATION_META)
            continue;
        const SegMeta* sm = (const SegMeta*) um->user_meta_data;
        if (!sm || !sm->class_map || sm->width == 0 || sm->height == 0) continue;
        if (sm->classes == 0 || sm->classes > (guint) kMaxSegClasses) continue;
        // GEOMETRY IS REFUSED, NOT TRUSTED -- the same rule mask() applies to
        // mask_params, and this was the one place in this file that did not
        // apply it. class_map is walked with a strided read whose extent is
        // width*height; a meta declaring dimensions no class map could have
        // turns that walk into an unbounded read of unmapped memory. Measured:
        // a SegMeta declaring 100000x100000 behind a 64-int buffer SEGFAULTS
        // without this line. An 8K class map (7680x4320) still fits, so this
        // rejects only what could not be real. The multiply is done in 64 bits
        // because two 32-bit dimensions overflow a guint.
        if ((uint64_t) sm->width * (uint64_t) sm->height > (uint64_t) kMaxSegCells)
            continue;
        got[n++] = sm;
    }
    if (n == 0) return;

    // PASS 2 -- REDUCE AND EMIT, oldest publisher first.
    out += ", \"segmentation\": [";
    std::vector<uint64_t> hist;
    for (size_t k = 0; k < n; ++k) {
        const SegMeta* sm = got[n - 1 - k];
        hist.assign((size_t) sm->classes, 0u);
        uint64_t other = 0, sampled = 0;
        for (guint y = 0; y < sm->height; y += (guint) kSegStride) {
            const gint* row = sm->class_map + (uint64_t) y * sm->width;
            for (guint x = 0; x < sm->width; x += (guint) kSegStride) {
                const gint c = row[x];
                ++sampled;
                // -1 is legitimate: nvinfer initialises every cell to -1 and
                // only overwrites it where some class beat the threshold.
                if (c >= 0 && (guint) c < sm->classes) ++hist[(size_t) c];
                else ++other;
            }
        }

        if (k) out += ", ";
        appendf(out, "{\"unique_id\": %d, \"width\": %u, \"height\": %u,"
                     " \"classes\": %u, \"stride\": %d, \"sampled\": %" PRIu64
                     ", \"coverage\": [",
                sm->unique_id, sm->width, sm->height, sm->classes,
                (int) kSegStride, sampled);
        size_t shown = 0, omitted = 0;
        for (guint c = 0; c < sm->classes; ++c) {
            if (hist[(size_t) c] == 0) continue;   // classes absent this frame
            if (shown >= (size_t) kMaxSegEmit) { ++omitted; continue; }
            appendf(out, "%s{\"class_id\": %u, \"fraction\": %.4f}",
                    shown ? ", " : "", c, (double) hist[(size_t) c] / (double) sampled);
            ++shown;
        }
        out += ']';
        if (omitted) appendf(out, ", \"classes_omitted\": %" PRIu64, (uint64_t) omitted);
        if (other) appendf(out, ", \"unclassified\": %.4f",
                           (double) other / (double) sampled);
        out += '}';
    }
    out += ']';
}

// -- shapes[] / fields{} : what a custom render .so decoded ------------------
//
// The same typed-shape vocabulary the VMS console already renders, so ONE
// producer feeds the console, the .json and the MQTT topic. A pipeline with no
// render stage, or a frame nobody published on, emits nothing.
//
// `width`/`height` are this record's pixel space; the payload records its own,
// and points are rescaled when they differ -- a render stage upstream of
// nvmultistreamtiler saw a different surface from a sink downstream of it. So
// every coordinate in a frame record (bbox, mask outline, shape) is in one
// space.
#ifdef FRAME_JSON_WITH_SHAPES
inline void shapes(std::string& out, NvDsFrameMeta* fm, int width, int height) {
    std::vector<render_meta::View> views;
    render_meta::views_on_frame(fm, views);
    if (views.empty()) return;

    bool any = false;
    for (size_t v = 0; v < views.size(); ++v) {
        const render_meta::View& vw = views[v];
        const double sx = vw.scale_x(width);
        const double sy = vw.scale_y(height);
        for (uint32_t i = 0; i < vw.n_shapes(); ++i) {
            const render_meta::Shape& s = vw.shape(i);
            if (!any) { out += ", \"shapes\": ["; any = true; }
            else out += ", ";
            appendf(out, "{\"kind\": \"%s\"", render_meta::kind_name(s.kind));

            if (s.kind == (uint32_t) render_meta::KIND_BOX) {
                // x/y/w/h, NOT the bbox{} object objects[] uses. shapes[] is
                // the VMS overlay contract, and the point of reusing it is that
                // ONE schema arrives over the socket, in the .json and on the
                // MQTT topic -- a reader of shapes[] must not have to know
                // which transport it came from. objects[] keeps bbox{}.
                appendf(out, ", \"x\": %.1f, \"y\": %.1f, \"w\": %.1f, \"h\": %.1f",
                        safe_num((double) s.x * sx), safe_num((double) s.y * sy),
                        safe_num((double) s.w * sx), safe_num((double) s.h * sy));
            } else {
                const render_meta::Point* pts = NULL;
                const uint32_t np = vw.points_of(s, &pts);
                out += ", \"points\": [";
                for (uint32_t p = 0; p < np; ++p)
                    appendf(out, "%s[%.1f, %.1f]", p ? ", " : "",
                            safe_num((double) pts[p].x * sx),
                            safe_num((double) pts[p].y * sy));
                out += ']';
                if (s.kind == (uint32_t) render_meta::KIND_SKELETON) {
                    const render_meta::Link* lk = NULL;
                    const uint32_t nl = vw.links_of(s, &lk);
                    out += ", \"links\": [";
                    for (uint32_t e = 0; e < nl; ++e)
                        appendf(out, "%s[%d, %d]", e ? ", " : "",
                                (int) lk[e].a, (int) lk[e].b);
                    out += ']';
                }
            }

            // The same optional tail, in the same order, as shape_json::tail().
            const char* label = vw.str(s.label_off);
            if (*label) { out += ", \"label\": "; escape(out, label); }
            if (s.id >= 0) appendf(out, ", \"id\": %" PRId64, (int64_t) s.id);
            if (s.score > 0.0f) appendf(out, ", \"score\": %.4f", safe_num((double) s.score));
            const char* text = vw.str(s.text_off);
            if (*text) { out += ", \"text\": "; escape(out, text); }
            out += '}';
        }
    }
    if (any) out += ']';

    // Free-form scalars a render published: "depth_m", "speed_kph". The
    // consumer displays them and never interprets them, which is what keeps
    // them usable by a pipeline it has never seen.
    bool anyf = false;
    for (size_t v = 0; v < views.size(); ++v) {
        const render_meta::View& vw = views[v];
        for (uint32_t i = 0; i < vw.n_fields(); ++i) {
            const render_meta::Field& f = vw.field(i);
            const char* key = vw.str(f.key_off);
            if (!*key) continue;
            if (!anyf) { out += ", \"fields\": {"; anyf = true; }
            else out += ", ";
            escape(out, key);
            if (f.type == (uint32_t) render_meta::FIELD_I)
                appendf(out, ": %" PRId64, (int64_t) f.ival);
            else if (f.type == (uint32_t) render_meta::FIELD_S) {
                out += ": ";
                escape(out, vw.str(f.sval_off));
            } else
                appendf(out, ": %.3f", safe_num(f.dval));
        }
    }
    if (anyf) out += '}';
}
#else
inline void shapes(std::string&, NvDsFrameMeta*, int, int) {}
#endif

// Native analytics is optional, like render shapes. The helper is public so
// MQTT's empty-frame gate and selective routes can preserve zero-object counts.
#ifdef FRAME_JSON_WITH_ANALYTICS
inline bool has_analytics_frame(const NvDsFrameMeta* fm) { return analytics_meta::has_frame(fm); }
inline bool has_analytics_object(const NvDsObjectMeta* ob) { return analytics_meta::has_object(ob); }
inline void analytics(std::string& out, const NvDsFrameMeta* fm) { analytics_meta::append_frame(out, fm); }
inline void object_analytics(std::string& out, const NvDsObjectMeta* ob) { analytics_meta::append_object(out, ob); }
#else
inline bool has_analytics_frame(const NvDsFrameMeta*) { return false; }
inline bool has_analytics_object(const NvDsObjectMeta*) { return false; }
inline void analytics(std::string&, const NvDsFrameMeta*) {}
inline void object_analytics(std::string&, const NvDsObjectMeta*) {}
#endif

// -- the record --------------------------------------------------------------

inline void object(std::string& out, NvDsObjectMeta* ob) {
    const NvOSD_RectParams& r = ob->rect_params;
    out += '{';
    // UNTRACKED_OBJECT_ID means no tracker upstream, or an object it has not
    // confirmed yet. Emitting it as-is would put 18446744073709551615 in the
    // payload, which most JSON readers silently turn into a float and mangle.
    if (ob->object_id == UNTRACKED_OBJECT_ID) out += "\"id\": null";
    else appendf(out, "\"id\": %" PRIu64, (uint64_t) ob->object_id);

    appendf(out, ", \"class_id\": %d, \"label\": ", ob->class_id);
    escape(out, ob->obj_label);
    appendf(out, ", \"confidence\": %.4f, \"bbox\": {\"left\": %.1f, \"top\": %.1f,"
                 " \"width\": %.1f, \"height\": %.1f}",
            ob->confidence, r.left, r.top, r.width, r.height);
    if (ob->tracker_confidence >= 0.0f && ob->tracker_confidence <= 1.0f)
        appendf(out, ", \"tracker_confidence\": %.4f", ob->tracker_confidence);
    // Both append nothing at all when the object has no such metadata, which is
    // every object a detector produces.
    attributes(out, ob);
    mask(out, ob);
    object_analytics(out, ob);
    out += '}';
}

// `width`/`height` are the coordinate space rect_params live in, i.e. the
// nvstreammux output size -- NOT the source resolution. The caller supplies it
// because only the caller knows: nvcustomprocess is handed the surface size,
// while another transport may have to be told.
inline void frame(std::string& out, NvDsFrameMeta* fm, int width, int height,
                  const std::vector<NvDsObjectMeta*>& objs) {
    appendf(out, "{\"frame\": %u, \"source_id\": %u, \"pts_ns\": %" PRIu64
                 ", \"width\": %d, \"height\": %d, \"objects\": [",
            fm->frame_num, fm->source_id, (uint64_t) fm->buf_pts, width, height);
    for (size_t i = 0; i < objs.size(); ++i) {
        if (i) out += ", ";
        object(out, objs[i]);
    }
    // Splitting the old `out += "]}"` in two appends the identical two bytes;
    // both helpers append nothing when the frame carries no such metadata.
    out += ']';
    segmentation(out, fm);
    shapes(out, fm, width, height);
    analytics(out, fm);
    out += '}';
}

// Objects on a frame that clear the confidence gate, in metadata order.
inline std::vector<NvDsObjectMeta*> objects_of(NvDsFrameMeta* fm, double min_conf) {
    std::vector<NvDsObjectMeta*> objs;
    for (NvDsMetaList* l = fm->obj_meta_list; l; l = l->next) {
        NvDsObjectMeta* ob = (NvDsObjectMeta*) l->data;
        if (ob && ob->confidence >= min_conf) objs.push_back(ob);
    }
    return objs;
}

// An object whose payload is NOT in its bbox: a classifier result, or an
// instance mask. nvinfer never assigns the classification placeholder's
// confidence, so it reads 0.0f and any gate above zero drops it -- and every
// attribute hanging off it. A sink that filters on confidence should OR this
// into that gate:
//
//     if (ob && (ob->confidence >= g_min_conf || frame_json::has_payload(ob)))
//
// This is what keeps a classification pipeline working on the VMS path, whose
// VMS_MIN_CONF defaults to 0.3, and on any JSON/MQTT pipeline whose operator
// raised JSON_MIN_CONF above zero.
inline bool has_payload(const NvDsObjectMeta* ob) {
    return ob && (ob->classifier_meta_list != NULL || ob->mask_params.data != NULL
                  || has_analytics_object(ob));
}

}  // namespace frame_json

#endif  // FRAME_JSON_H_
