// render_meta.h -- how a generated render .so hands over what it decoded.
//
// THE GAP THIS CLOSES. A network-type=100 stage (pose, lane, depth, anything
// custom) produces a raw tensor, and the generated render .so is the ONLY code
// in the pipeline that knows its layout. Today it decodes keypoints, draws them
// with OpenCV, and returns -- everything it worked out dies with the stack
// frame. Every sink downstream sees an empty frame and cannot tell "the model
// found nothing" from "nobody published anything".
//
// THE CONTRACT. One NvDsUserMeta per frame carrying TYPED SHAPES in exactly the
// vocabulary the VMS console already renders (parsers/shape_json.h):
//
//     box       x/y/w/h                     -- a thing with an extent
//     polygon   closed ring of points       -- drivable area, mask outline
//     polyline  open path of points         -- a lane, a trajectory
//     skeleton  points + links between them -- a pose
//     fields    named scalars               -- "depth_m": 12.4
//
// The vocabulary is CLOSED on purpose. Nothing here knows what a keypoint
// means, which is exactly what lets a console, a .json and an MQTT topic that
// understand four primitives keep working for a pipeline they have never seen.
// A fifth kind would mean changing every consumer, so express new things with
// the four.
//
//     // ---- producer: inside the generated render .so ----
//     render_meta::Builder sb(in->W, in->H);            // PIXELS, frame space
//     render_meta::Skeleton sk(NUM_KEYPOINTS);          // 18 slots, all absent
//     for (int c = 0; c < NUM_KEYPOINTS; ++c)
//         if (found[c]) sk.set(c, x_px, y_px);          // fill what we found
//     for (int k = 0; k < NUM_LIMBS; ++k)
//         sk.link(TOPOLOGY[k].kpt_a, TOPOLOGY[k].kpt_b);
//     sb.skeleton(sk, "person", track_id, score);
//     std::vector<double> xy;                           // x0,y0,x1,y1,...
//     sb.polyline(xy, "lane");                          // PASS THE CONTAINER --
//     sb.polygon (xy, "drivable");                      //   the point count is
//     sb.field_d("depth_m", 12.4);                      //   derived from it
//     sb.publish(in->frame_meta);                       // once per frame
//
//     // ---- consumer: inside a sink ----
//     std::vector<render_meta::View> vs;
//     render_meta::views_on_frame(fm, vs);              // publish order, valid
//     render_meta::to_shape_line(vs[i], line, ox, oy);  // 1:1 onto shape_json
//
// COORDINATES ARE PIXELS in the space the publisher declares, normally the
// nvstreammux output size -- the same space NvDsObjectMeta::rect_params live in
// and the same numbers frame_json.h reports as `width`/`height`, which for an
// nvcustomprocess library is exactly in->W / in->H. NOT normalised 0..1: every
// sink formats points with "%.1f", so normalised coordinates collapse every
// joint of a pose onto the same point. A consumer whose own pixel space differs
// (anything downstream of nvmultistreamtiler) rescales with View::scale_x/y.
//
// ---------------------------------------------------------------------------
// THREAT MODEL: THE PRODUCER IS UNTRUSTED.
//
// The .so that calls this is LLM-authored and compiled without review. It will
// get counts wrong, pass null arrays, hand over NaN coordinates and a
// half-filled label buffer. None of that may be allowed to corrupt the pipeline
// or to poison a document that a whole run's output depends on, so:
//
//   * Every append is validated BEFORE the first store, so a rejected shape
//     leaves the payload exactly as it was -- a half-written shape is the one
//     thing a consumer cannot detect. Out-of-range link indices are dropped
//     individually, because one bad entry in a 21-limb topology should cost
//     that limb, not the skeleton.
//   * Coordinates must be finite and within COORD_LIMIT. "%.1f" of 1e300 is 300
//     digits, which overruns the 512-byte buffer every JSON writer in this tree
//     formats through, and "%.4f" of a NaN is `nan` -- a token no JSON parser
//     accepts, which would destroy the entire run's output rather than one key.
//   * Strings are bounded and UTF-8-validated ONCE, here, at the boundary where
//     untrusted text enters. A labels file in Latin-1 otherwise puts raw
//     0x80..0xFF bytes into every downstream document and json.loads() raises
//     UnicodeDecodeError on the whole file.
//   * open() re-validates everything a consumer will read -- magic, version,
//     every count, every per-shape slice, pool termination, every coordinate --
//     so a sink that checks only View::valid() cannot be made to read out of
//     bounds by a payload this header did not build.
//   * THE WRITER'S OWN ARGUMENTS ARE UNTRUSTED TOO. A path is x0,y0,x1,y1,...,
//     so a POINT count is HALF the number of doubles, and `sb.polyline(&xy[0],
//     xy.size(), "lane")` used to read one whole buffer past the end -- inside
//     this header, on the streaming thread, in a .so built with no -Wall and no
//     _FORTIFY_SOURCE. The counts therefore no longer travel on their own: pass
//     a container and the count is DERIVED, pass a container AND a count and the
//     count is CLAMPED to what the container holds, and a bare pointer must be
//     paired with NPoints(n) / NLinks(n). See "-- writing --" below; the old
//     `pointer, size_t` call no longer compiles.
//
// ---------------------------------------------------------------------------
// MEMORY. Three properties, none of them optional:
//
//   1. ONE user meta per frame carrying an ARRAY of shapes, never one meta per
//      shape. nvds_add_user_meta_to_frame PREPENDS (it passes flag=0 to
//      nvds_add_meta_to_parent), so N metas arrive in reverse and cost N
//      acquisitions, N copies and N releases. One meta has one order and one
//      allocation.
//   2. ONE FLAT ALLOCATION. The payload is self-describing, with every
//      variable-length part (points, links, strings) inside it and addressed by
//      byte offset. copy_func is therefore one memcpy of total_bytes and
//      release_func is one g_free: there is no deep-copy tree to get wrong and
//      no pointer inside the payload that can dangle. (NVIDIA's own
//      segmentation-meta pair in gstnvinfer_meta_utils.cpp g_malloc's and then
//      `delete`s; a flat blob makes that class of bug unrepresentable.)
//   3. THE PUBLISHER IS PINNED IN MEMORY. This is the one that actually crashes
//      pipelines. copy_func/release_func are stored as raw pointers INSIDE the
//      metadata and called much later, at nvds_clear_meta_list time -- while
//      the nvcustomprocess element dlclose()s the custom .so from its
//      GstBaseTransform::stop, i.e. on the PAUSED->READY transition, with
//      frames still in flight. Releasing that metadata afterwards jumps into an
//      unmapped page: a SIGSEGV at teardown, far from the code that caused it.
//      publish() therefore re-dlopen()s its own object with
//      RTLD_NOLOAD|RTLD_NODELETE and leaks the handle on purpose, so the
//      element's dlclose() decrements a refcount that never reaches zero. The
//      same pin keeps the descriptor string alive: nvds_get_user_meta_type is
//      g_quark_from_static_string(s) + NVDS_START_USER_META, and GLib stores
//      that char* WITHOUT copying it. Define RENDER_META_NO_PIN to opt out.
//
// The meta TYPE is resolved at runtime by both sides and must never be
// hardcoded, serialised or persisted: the same descriptor yields a different
// integer in a different process, because it depends on how many quarks were
// interned first.
//
// THREAD SAFETY. A Builder is a plain local object; build one per frame on the
// stack. publish() touches only the batch meta pool, which is what every other
// DeepStream producer does from a streaming thread. A published payload is
// immutable, so any number of threads may have it copied concurrently.

#ifndef RENDER_META_H_
#define RENDER_META_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <dlfcn.h>
#include <glib.h>

#include "nvdsmeta.h"

namespace render_meta {

// The descriptor lives in a header so every module that registers it compiles
// in its own copy of the literal, mirroring nvdscustomusermeta.h's
// NVIDIA.USER.CUSTOM_META.
#define RENDER_META_DESCRIPTOR "PIPEGEN.RENDER.SHAPES"

// -- capacities --------------------------------------------------------------
//
// These bound the payload, every loop that reads it, and the JSON one frame can
// produce. They are generous for real overlay data (8 people x 18 keypoints is
// 144 points) and exist so that a runaway decoder costs a truncated key rather
// than a gigabyte of JSON or a stalled streaming thread. Worst-case payload is
// about 37 KB.
enum {
    MAGIC       = 0x50475348u,   // 'PGSH'
    VERSION     = 1u,
    MAX_SHAPES  = 64,
    MAX_POINTS  = 2048,          // across all shapes in one frame
    MAX_LINKS   = 1024,
    MAX_FIELDS  = 32,
    MAX_STRING  = 64,            // bytes per label/text/key, NUL included
    MAX_POOL    = 8192           // total string-pool bytes
};

// Pixels. Anything larger is a decode bug and is refused rather than stored.
static const double COORD_LIMIT = 1.0e6;

enum Kind {
    KIND_BOX      = 0,   // axis-aligned rectangle; uses x/y/w/h
    KIND_POLYGON  = 1,   // closed ring of points
    KIND_POLYLINE = 2,   // open path of points
    KIND_SKELETON = 3,   // points plus the edges joining them
    KIND__COUNT   = 4
};

enum FieldType { FIELD_D = 0, FIELD_I = 1, FIELD_S = 2 };

// The four names the console knows. An unrecognised kind degrades to "box"
// rather than inventing one nothing downstream can draw -- though open() also
// rejects such a payload outright, so this is belt and braces.
inline const char* kind_name(uint32_t k) {
    switch (k) {
        case KIND_POLYGON:  return "polygon";
        case KIND_POLYLINE: return "polyline";
        case KIND_SKELETON: return "skeleton";
        default:            return "box";
    }
}

// -- the wire layout ---------------------------------------------------------
//
//   [ Header | Shape[] | Point[] | Link[] | Field[] | string pool ]
//
// Sections appear in that order, each a whole number of 8-byte units long, so
// every section starts 8-aligned inside one g_malloc'd block. All offsets are
// byte offsets from the start of the blob. A string offset of 0 means "absent":
// the pool always begins with a NUL, so 0 can never name a real string.

struct Header {
    uint32_t magic;
    uint32_t version;
    uint32_t total_bytes;    // copy_func memcpy's exactly this many bytes
    uint32_t width;          // the PIXEL space every point below lives in
    uint32_t height;
    uint32_t n_shapes;
    uint32_t off_shapes;
    uint32_t n_points;
    uint32_t off_points;
    uint32_t n_links;
    uint32_t off_links;
    uint32_t n_fields;
    uint32_t off_fields;
    uint32_t off_strings;
    uint32_t strings_bytes;
    uint32_t producer;       // free-form stage id; 0 when unset
};

struct Shape {
    int64_t  id;             // track id, or -1 for "none"
    float    score;          // <= 0 means "none"
    float    x, y, w, h;     // KIND_BOX only; zero otherwise
    uint32_t kind;
    uint32_t first_point, n_points;
    uint32_t first_link,  n_links;
    uint32_t label_off, text_off;
};

struct Point { float x, y; };
struct Link  { int32_t a, b; };

struct Field {
    double   dval;
    int64_t  ival;
    uint32_t type;
    uint32_t key_off;
    uint32_t sval_off;
    uint32_t reserved;
};

// The blob is memcpy'd across a dlopen boundary between a .so built at
// -std=c++17 and sinks built at -std=c++14, so both must agree on the layout
// exactly. A mismatch fails the build instead of corrupting data.
static_assert(sizeof(Header) == 64, "render_meta::Header layout changed");
static_assert(sizeof(Shape)  == 56, "render_meta::Shape layout changed");
static_assert(sizeof(Point)  == 8,  "render_meta::Point layout changed");
static_assert(sizeof(Link)   == 8,  "render_meta::Link layout changed");
static_assert(sizeof(Field)  == 32, "render_meta::Field layout changed");
static_assert(sizeof(Header) % 8 == 0 && sizeof(Shape) % 8 == 0 &&
              sizeof(Point)  % 8 == 0 && sizeof(Link)  % 8 == 0 &&
              sizeof(Field)  % 8 == 0,
              "every render_meta section must keep the next one 8-aligned");

namespace detail {

// Hidden visibility is load-bearing, not decoration.
//
// The address stored in the metadata must be THIS module's copy: without it an
// inline function's vague-linkage symbol can be interposed across shared
// objects and the pin below would pin the wrong library. Worse, GCC gives a
// plain inline function's local statics STB_GNU_UNIQUE binding, which the
// dynamic linker unifies process-wide -- so with two render libraries the
// second would observe the first's `pinned` flag and never pin ITSELF. Hidden
// visibility demotes that guard to module-local BSS.
#define RENDER_META_LOCAL __attribute__((visibility("hidden")))

RENDER_META_LOCAL inline void module_anchor() {}

#ifndef RENDER_META_NO_PIN
// RTLD_NOLOAD can never load anything new: it only marks what is already
// resident as non-unloadable. One dlopen per module; the handle is deliberately
// leaked, because the leak IS the pin. (Expect LeakSanitizer to report it.)
RENDER_META_LOCAL inline bool pin_publisher_library() {
    static const bool pinned = []() -> bool {
        Dl_info info;
        std::memset(&info, 0, sizeof(info));
        if (dladdr((void*) &module_anchor, &info) &&
            info.dli_fname && *info.dli_fname) {
            void* h = dlopen(info.dli_fname,
                             RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
            if (h) return true;
        }
        return false;   // e.g. compiled into the executable, which is never
                        // dlclose'd anyway
    }();
    return pinned;
}
#else
RENDER_META_LOCAL inline bool pin_publisher_library() { return false; }
#endif

// "data" is the NvDsUserMeta*, not the payload -- see
// deepstream_user_metadata_app.c:69. One contiguous block in, one out.
//
// A NULL copy_func would make the payload vanish silently the first time the
// batch is copied (nvstreamdemux, nvmsgconv, a secondary branch); a NULL
// release_func would leak one payload per frame, forever, with no warning.
// Neither failure produces a diagnostic, which is why both are always set.
RENDER_META_LOCAL inline gpointer copy_shapes_meta(gpointer data, gpointer) {
    NvDsUserMeta* um = (NvDsUserMeta*) data;
    if (!um || !um->user_meta_data) return NULL;
    const Header* h = (const Header*) um->user_meta_data;
    if (h->magic != (uint32_t) MAGIC || h->total_bytes < sizeof(Header)) return NULL;
    void* dst = g_try_malloc((gsize) h->total_bytes);
    if (!dst) return NULL;
    std::memcpy(dst, um->user_meta_data, h->total_bytes);
    return (gpointer) dst;
}

RENDER_META_LOCAL inline void release_shapes_meta(gpointer data, gpointer) {
    NvDsUserMeta* um = (NvDsUserMeta*) data;
    if (um && um->user_meta_data) {
        g_free(um->user_meta_data);
        um->user_meta_data = NULL;   // a second release must be a no-op
    }
}

// A section is 8-aligned, inside the block, and cannot overflow when the
// element count is multiplied out.
inline bool fits(uint32_t off, uint32_t n, size_t elem, uint64_t total) {
    if (n == 0) return true;
    if ((off & 7u) != 0u) return false;
    const uint64_t end = (uint64_t) off + (uint64_t) n * (uint64_t) elem;
    return (uint64_t) off < total && end <= total;
}

inline bool finite_coord(double v) {
    return std::isfinite(v) && v >= -COORD_LIMIT && v <= COORD_LIMIT;
}

// Bounded, UTF-8-validating copy. Invalid sequences become U+FFFD; the result
// is always NUL-terminated and never longer than MAX_STRING-1 bytes.
//
// Validation happens HERE, once, at the boundary where untrusted producer text
// enters -- which is what makes every consumer safe, including shape_json.h,
// whose escape() passes bytes through unexamined.
//
// THE SCAN IS BOUNDED BY MAX_STRING, not by a NUL. A `const char*` carries no
// length, and a producer's buffer is not guaranteed to be terminated -- a
// half-filled fixed array is the normal untrusted case, not an exotic one.
// Scanning for a terminator that is not there reads off the end of the
// caller's object; AddressSanitizer caught exactly that here, on a
// deliberately non-terminated char[64]. MAX_STRING-1 is the smallest bound
// that can still fill the field, so it is the least this can read. Callers
// must still pass a buffer of at least MAX_STRING bytes or a terminated
// string; nothing in the C string contract lets this check that for them.
inline void sanitize(const char* src, std::vector<char>& out) {
    out.clear();
    if (!src) return;
    const unsigned char* p = (const unsigned char*) src;
    size_t n = 0;
    while (n + 1u < (size_t) MAX_STRING && p[n]) ++n;
    size_t i = 0;
    while (i < n && out.size() + 4u < (size_t) MAX_STRING) {
        const unsigned char c = p[i];
        if (c < 0x80u) { out.push_back((char) c); ++i; continue; }
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
        if (ok) { for (size_t k = 0; k <= need; ++k) out.push_back((char) p[i + k]); i += need + 1; }
        else    { out.push_back((char) 0xEFu); out.push_back((char) 0xBFu);
                  out.push_back((char) 0xBDu); ++i; }
    }
}

}  // namespace detail

// Resolved once PER MODULE, on first use. Producer and consumer agree because
// there is one libnvds_meta in the process and therefore one quark table.
//
// Pinned first, because interning the descriptor stores its ADDRESS: whichever
// module donates the literal must outlive the process-global quark table, and
// the pin runs unconditionally before the static so every caller protects
// itself whether or not it is the one that interns.
//
// RENDER_META_LOCAL is load-bearing here for the same reason it is on the pin.
// A plain inline function's local static gets STB_GNU_UNIQUE binding, which the
// dynamic linker unifies process-wide -- verified with `nm`, which reported `u`
// before this was hidden and `b` after. Unified, only the first module would
// run the initialiser, which quietly couples two modules' lifetimes through a
// variable neither of them can see. Hidden, each module keeps its own copy and
// the behaviour no longer depends on link order, on GCC's unique-symbol
// extension, or on -fno-gnu-unique.
RENDER_META_LOCAL inline NvDsMetaType meta_type() {
    detail::pin_publisher_library();
    static const NvDsMetaType t =
        nvds_get_user_meta_type((gchar*) RENDER_META_DESCRIPTOR);
    return t;
}

// -- reading -----------------------------------------------------------------

// A validated, read-only window onto one published blob. Every accessor is
// bounds-checked against the header, so a truncated, stale or forged payload
// yields empty results instead of a fault.
struct View {
    const Header* h;
    const Shape*  sh;
    const Point*  pt;
    const Link*   ln;
    const Field*  fd;
    const char*   sp;
    uint32_t      sp_bytes;

    View() : h(NULL), sh(NULL), pt(NULL), ln(NULL), fd(NULL), sp(NULL),
             sp_bytes(0) {}

    bool     valid()    const { return h != NULL; }
    uint32_t width()    const { return h ? h->width  : 0u; }
    uint32_t height()   const { return h ? h->height : 0u; }
    uint32_t producer() const { return h ? h->producer : 0u; }
    uint32_t n_shapes() const { return h ? h->n_shapes : 0u; }
    uint32_t n_fields() const { return h ? h->n_fields : 0u; }

    const Shape& shape(uint32_t i) const { return sh[i]; }
    const Field& field(uint32_t i) const { return fd[i]; }

    // Offset 0, and anything past the pool, both mean "absent". The pool's last
    // byte is verified NUL by open(), so what comes back is always a valid,
    // UTF-8-clean C string.
    const char* str(uint32_t off) const {
        return (sp && off != 0u && off < sp_bytes) ? (sp + off) : "";
    }

    uint32_t points_of(const Shape& s, const Point** out) const {
        *out = NULL;
        if (!pt || s.n_points == 0u) return 0u;
        *out = pt + s.first_point;          // slice checked by open()
        return s.n_points;
    }

    uint32_t links_of(const Shape& s, const Link** out) const {
        *out = NULL;
        if (!ln || s.n_links == 0u) return 0u;
        *out = ln + s.first_link;
        return s.n_links;
    }

    // Map this payload's pixel space onto a consumer's. Returns exactly 1.0
    // when the spaces match or the publisher declared none, so the common case
    // is a bit-exact identity rather than a rounded multiply. A sink downstream
    // of nvmultistreamtiler sees a different surface from the render stage.
    double scale_x(int target_w) const {
        if (!h || h->width == 0u || target_w <= 0) return 1.0;
        if ((uint32_t) target_w == h->width) return 1.0;
        return (double) target_w / (double) h->width;
    }
    double scale_y(int target_h) const {
        if (!h || h->height == 0u || target_h <= 0) return 1.0;
        if ((uint32_t) target_h == h->height) return 1.0;
        return (double) target_h / (double) h->height;
    }
};

// Validate a blob and return a window onto it. An invalid blob gives
// View::valid() == false and every caller simply skips it -- a sink must not go
// quiet because one generated .so wrote nonsense.
//
// After this returns a valid View, every slice is in range, every kind is
// known, every string offset resolves inside a NUL-terminated pool and every
// coordinate is a finite pixel value. Consumers need no further checks.
inline View open(const void* blob) {
    View v;
    if (!blob) return v;
    const Header* h = (const Header*) blob;
    if (h->magic != (uint32_t) MAGIC || h->version != (uint32_t) VERSION) return v;
    if (h->total_bytes < sizeof(Header)) return v;
    if (h->n_shapes > (uint32_t) MAX_SHAPES) return v;
    if (h->n_points > (uint32_t) MAX_POINTS) return v;
    if (h->n_links  > (uint32_t) MAX_LINKS)  return v;
    if (h->n_fields > (uint32_t) MAX_FIELDS) return v;

    const uint64_t total = (uint64_t) h->total_bytes;
    if (!detail::fits(h->off_shapes, h->n_shapes, sizeof(Shape), total)) return v;
    if (!detail::fits(h->off_points, h->n_points, sizeof(Point), total)) return v;
    if (!detail::fits(h->off_links,  h->n_links,  sizeof(Link),  total)) return v;
    if (!detail::fits(h->off_fields, h->n_fields, sizeof(Field), total)) return v;
    if ((uint64_t) h->off_strings + (uint64_t) h->strings_bytes > total) return v;

    const char* base = (const char*) blob;
    if (h->strings_bytes != 0u &&
        base[(size_t) h->off_strings + h->strings_bytes - 1u] != '\0') return v;

    const Shape* sh = h->n_shapes ? (const Shape*) (base + h->off_shapes) : NULL;
    for (uint32_t i = 0; i < h->n_shapes; ++i) {
        const Shape& s = sh[i];
        if (s.kind >= (uint32_t) KIND__COUNT) return v;
        // Subtraction, never addition: the left operand is already bounded, so
        // this cannot overflow.
        if (s.first_point > h->n_points) return v;
        if (s.n_points > h->n_points - s.first_point) return v;
        if (s.first_link > h->n_links) return v;
        if (s.n_links > h->n_links - s.first_link) return v;
        if (s.kind != (uint32_t) KIND_SKELETON && s.n_links != 0u) return v;
        if (s.kind != (uint32_t) KIND_BOX && s.n_points == 0u) return v;
        if (!std::isfinite((double) s.score)) return v;
        if (s.kind == (uint32_t) KIND_BOX &&
            !(detail::finite_coord((double) s.x) && detail::finite_coord((double) s.y) &&
              detail::finite_coord((double) s.w) && detail::finite_coord((double) s.h)))
            return v;
    }
    const Point* pt = h->n_points ? (const Point*) (base + h->off_points) : NULL;
    for (uint32_t i = 0; i < h->n_points; ++i)
        if (!detail::finite_coord((double) pt[i].x) ||
            !detail::finite_coord((double) pt[i].y)) return v;

    const Field* fd = h->n_fields ? (const Field*) (base + h->off_fields) : NULL;
    for (uint32_t i = 0; i < h->n_fields; ++i)
        if (fd[i].type > (uint32_t) FIELD_S || !std::isfinite(fd[i].dval)) return v;

    v.h  = h;
    v.sh = sh;
    v.pt = pt;
    v.ln = h->n_links ? (const Link*) (base + h->off_links) : NULL;
    v.fd = fd;
    v.sp = h->strings_bytes ? (base + h->off_strings) : NULL;
    v.sp_bytes = h->strings_bytes;
    return v;
}

namespace detail {

// Shared by views_on_frame / views_on_object. The list walk is capped so a
// corrupted or cyclic list cannot hang the streaming thread.
inline void views_of_list(NvDsMetaList* list, std::vector<View>& out) {
    const NvDsMetaType want = meta_type();
    const size_t first = out.size();
    size_t walked = 0;
    for (NvDsMetaList* l = list; l && walked < 4096u; l = l->next) {
        ++walked;
        NvDsUserMeta* um = (NvDsUserMeta*) l->data;
        if (!um || um->base_meta.meta_type != want || !um->user_meta_data) continue;
        View v = open(um->user_meta_data);
        if (v.valid()) out.push_back(v);
    }
    // The list is built with g_list_prepend, so the walk above returned the
    // LAST publisher first. Two render stages on one pipeline must not swap
    // places from frame to frame.
    //
    // CAVEAT, measured: nvds_copy_frame_user_meta_list re-prepends as it walks,
    // so after DeepStream copies a frame's user meta list the RELATIVE order of
    // two separate payloads flips. That is out of this header's hands. It never
    // affects the shapes WITHIN one payload -- they are an array whose order is
    // fixed at publish time and preserved by the memcpy -- and a single
    // publisher, which is the normal case, is unaffected entirely.
    std::reverse(out.begin() + (std::ptrdiff_t) first, out.end());
}

}  // namespace detail

// Every shapes payload on a frame, in PUBLISH order. Appends; does not clear.
inline void views_on_frame(NvDsFrameMeta* fm, std::vector<View>& out) {
    if (!fm || !fm->frame_user_meta_list) return;
    detail::views_of_list(fm->frame_user_meta_list, out);
}

// The same, for a payload published onto an object rather than the frame.
inline void views_on_object(NvDsObjectMeta* ob, std::vector<View>& out) {
    if (!ob || !ob->obj_user_meta_list) return;
    detail::views_of_list(ob->obj_user_meta_list, out);
}

// Is there anything to say about this frame at all? Sinks use this to decide
// whether to emit a key, so a frame nobody published on produces byte-identical
// output to a build that had never heard of shapes.
inline bool present(NvDsFrameMeta* fm) {
    std::vector<View> v;
    views_on_frame(fm, v);
    return !v.empty();
}

// -- writing -----------------------------------------------------------------
//
// HOW A COUNT REACHES THIS HEADER, AND WHY IT IS NOT A BARE size_t.
//
// Every path shape is a flat x0,y0,x1,y1,... buffer, so a POINT count is HALF
// the number of doubles -- and passing the array length where the point count
// belongs is the single mistake every producer of this API makes:
//
//     sb.polyline(&xy[0], xy.size(), "lane");   // 8 doubles -> read 16
//
// That call used to compile and read a whole buffer past the end (measured:
// AddressSanitizer heap-buffer-overflow, READ of size 8, 0 bytes to the right
// of a 64-byte region, inside Builder::path_). An interface a competent caller
// gets wrong is a defective interface, so the count no longer travels alone:
//
//   1. PASS THE CONTAINER and the count is DERIVED. Nothing to get wrong, and
//      it is the shortest form to write:
//          sb.polyline(xy, "lane");          // std::vector<double>, or double[N]
//          sb.polygon (pts, "drivable");     // std::vector<XY>,     or XY[N]
//   2. PASS A CONTAINER AND A COUNT and the count is CLAMPED to what the
//      container actually holds, so the mistake above now produces the RIGHT
//      shape (xy.size()/2 points) instead of a read off the end:
//          sb.polyline(xy, npoints, "lane");
//   3. A BARE POINTER carries no length, so the unit must be spelled at the
//      call site. `sb.polyline(ptr, n, "lane")` DOES NOT COMPILE; write
//          sb.polyline(ptr, render_meta::NPoints(n), "lane");
//      and `n` is unmistakably POINTS, not doubles.
//
// Links work the same way: `links` is 2*nlinks ints, a C `int[M][2]` topology
// counts as M links, and the raw form needs NLinks(n).

// A count of POINTS -- never of doubles. Explicit on purpose: it is what stops
// a bare xy.size() from sliding into a call this header cannot bounds-check.
struct NPoints {
    size_t n;
    explicit NPoints(size_t points) : n(points) {}
};

// A count of LINKS -- never of ints. The link array holds 2*n ints.
struct NLinks {
    size_t n;
    explicit NLinks(size_t links) : n(links) {}
};

// One point in the publisher's pixel space. A container of these cannot be
// miscounted, because its length IS the point count.
struct XY { double x, y; };

// A read-only view of a point sequence THAT KNOWS HOW LONG IT IS.
//
// It converts implicitly from every container a decoder already has, which is
// what makes the safe call the shortest one to write and lets ONE Builder
// overload serve std::vector<double>, double[N], std::vector<XY> and XY[N].
// The raw-pointer constructor is the only one that takes the caller's word for
// the length, and it cannot be reached without writing NPoints.
class Coords {
public:
    Coords(const std::vector<double>& xy)                      // x0,y0,x1,y1,...
        : m_d(xy.empty() ? NULL : &xy[0]), m_p(NULL), m_n(xy.size() / 2u) {}
    template <size_t N>
    Coords(const double (&xy)[N])
        : m_d(xy), m_p(NULL), m_n(N / 2u) {}
    Coords(const std::vector<XY>& pts)
        : m_d(NULL), m_p(pts.empty() ? NULL : &pts[0]), m_n(pts.size()) {}
    template <size_t N>
    Coords(const XY (&pts)[N])
        : m_d(NULL), m_p(pts), m_n(N) {}
    // npoints is POINTS, NOT DOUBLES: xy must hold 2*npoints doubles. This is
    // the one form nothing can check, which is why saying it takes an NPoints.
    Coords(const double* xy, NPoints npoints)
        : m_d(xy), m_p(NULL), m_n(xy ? npoints.n : 0u) {}

    size_t points() const { return m_n; }
    bool   empty()  const { return m_n == 0u || (m_d == NULL && m_p == NULL); }
    double x(size_t i) const { return m_d ? m_d[2u * i]      : m_p[i].x; }
    double y(size_t i) const { return m_d ? m_d[2u * i + 1u] : m_p[i].y; }

private:
    const double* m_d;   // flat, 2 doubles per point
    const XY*     m_p;   // or one XY per point; exactly one of the two is set
    size_t        m_n;   // POINTS
};

// The same idea for a skeleton's edges: 2 ints per link, or a C `int[M][2]`
// topology table, which is the shape a decoder usually already has.
class LinkList {
public:
    LinkList() : m_p(NULL), m_n(0u) {}
    LinkList(const std::vector<int>& links)                    // a0,b0,a1,b1,...
        : m_p(links.empty() ? NULL : &links[0]), m_n(links.size() / 2u) {}
    template <size_t M>
    LinkList(const int (&links)[M]) : m_p(links), m_n(M / 2u) {}
    template <size_t M>
    LinkList(const int (&links)[M][2]) : m_p(&links[0][0]), m_n(M) {}
    // nlinks is LINKS, NOT INTS: links must hold 2*nlinks ints.
    LinkList(const int* links, NLinks nlinks)
        : m_p(links), m_n(links ? nlinks.n : 0u) {}

    size_t count() const { return m_p ? m_n : 0u; }
    int    a(size_t i) const { return m_p[2u * i]; }
    int    b(size_t i) const { return m_p[2u * i + 1u]; }

private:
    const int* m_p;
    size_t     m_n;      // LINKS
};

// A pose model finds SOME keypoints of SOME people, and the wire format carries
// only the ones that were found -- so the link indices have to be renumbered
// onto the surviving subset. That renumbering is exactly the step a
// hand-written decoder gets wrong, so it is done here instead: fill by ORIGINAL
// keypoint index, declare the topology in ORIGINAL indices, and links whose
// endpoints were not both found are dropped and the rest renumbered.
class Skeleton {
public:
    explicit Skeleton(size_t nslots)
        : m_x(nslots, 0.0f), m_y(nslots, 0.0f), m_have(nslots, 0) {}

    size_t slots() const { return m_have.size(); }
    bool present(size_t i) const { return i < m_have.size() && m_have[i] != 0; }

    void set(size_t i, double x, double y) {
        if (i >= m_have.size()) return;
        m_x[i] = (float) x;
        m_y[i] = (float) y;
        m_have[i] = 1;
    }

    void link(size_t a, size_t b) {
        if (a >= m_have.size() || b >= m_have.size() || a == b) return;
        if (m_links.size() >= (size_t) MAX_LINKS * 2u) return;
        m_links.push_back((int32_t) a);
        m_links.push_back((int32_t) b);
    }

    size_t found() const {
        size_t n = 0;
        for (size_t i = 0; i < m_have.size(); ++i) if (m_have[i]) ++n;
        return n;
    }

    void compact(std::vector<float>& pts, std::vector<int32_t>& links) const {
        pts.clear();
        links.clear();
        std::vector<int32_t> remap(m_have.size(), -1);
        for (size_t i = 0; i < m_have.size(); ++i) {
            if (!m_have[i]) continue;
            remap[i] = (int32_t) (pts.size() / 2);
            pts.push_back(m_x[i]);
            pts.push_back(m_y[i]);
        }
        for (size_t k = 0; k + 1 < m_links.size(); k += 2) {
            const int32_t a = remap[(size_t) m_links[k]];
            const int32_t b = remap[(size_t) m_links[k + 1]];
            if (a < 0 || b < 0) continue;
            links.push_back(a);
            links.push_back(b);
        }
    }

private:
    std::vector<float>   m_x, m_y;
    std::vector<uint8_t> m_have;
    std::vector<int32_t> m_links;
};

// Accumulates shapes, then serialises them into ONE block and attaches it.
// Nothing is taken from the metadata pool until publish().
//
// The signatures deliberately mirror shape_json::Line's, parameter for
// parameter, so forwarding is a rename and a renderer cannot express a shape
// the console cannot draw.
class Builder {
public:
    Builder(int width, int height, uint32_t producer = 0)
        : m_w(width  > 0 ? (uint32_t) width  : 0u),
          m_h(height > 0 ? (uint32_t) height : 0u),
          m_producer(producer) {
        m_strings.push_back('\0');   // offset 0 is reserved for "absent"
    }

    bool box(double x, double y, double w, double h, const char* label,
             int64_t id = -1, double score = 0.0, const char* text = NULL) {
        if (m_shapes.size() >= (size_t) MAX_SHAPES) return false;
        if (!detail::finite_coord(x) || !detail::finite_coord(y) ||
            !detail::finite_coord(w) || !detail::finite_coord(h)) return false;
        Shape s;
        init_(s, KIND_BOX, label, id, score, text);
        s.x = (float) x; s.y = (float) y;
        s.w = (float) w; s.h = (float) h;
        m_shapes.push_back(s);
        return true;
    }

    // -- paths: polygon / polyline --------------------------------------------
    //
    // FORM 1 -- pass the container, the POINT count is derived from it. Accepts
    // std::vector<double> / double[N] (flat x0,y0,x1,y1,...) and
    // std::vector<XY> / XY[N]. Use this one.
    bool polygon(Coords xy, const char* label,
                 int64_t id = -1, double score = 0.0, const char* text = NULL) {
        return path_(KIND_POLYGON, xy, xy.points(), label, id, score, text);
    }
    // FORM 2 -- npoints is POINTS, NOT DOUBLES, and is CLAMPED to what `xy`
    // holds, so polygon(xy, xy.size(), ...) yields xy.size()/2 points instead
    // of reading past the end.
    bool polygon(Coords xy, size_t npoints, const char* label,
                 int64_t id = -1, double score = 0.0, const char* text = NULL) {
        return path_(KIND_POLYGON, xy, clamp_points_(npoints, xy),
                     label, id, score, text);
    }
    // FORM 3 -- a bare pointer knows no length, so the unit is spelled at the
    // call site: npoints is POINTS, NOT DOUBLES, and `xy` must hold 2*npoints
    // doubles. Nothing here can verify that, which is the whole reason this
    // overload demands NPoints(n) rather than accepting a bare count.
    bool polygon(const double* xy, NPoints npoints, const char* label,
                 int64_t id = -1, double score = 0.0, const char* text = NULL) {
        return path_(KIND_POLYGON, Coords(xy, npoints), npoints.n,
                     label, id, score, text);
    }

    bool polyline(Coords xy, const char* label,
                  int64_t id = -1, double score = 0.0, const char* text = NULL) {
        return path_(KIND_POLYLINE, xy, xy.points(), label, id, score, text);
    }
    // npoints is POINTS, NOT DOUBLES; clamped to what `xy` holds.
    bool polyline(Coords xy, size_t npoints, const char* label,
                  int64_t id = -1, double score = 0.0, const char* text = NULL) {
        return path_(KIND_POLYLINE, xy, clamp_points_(npoints, xy),
                     label, id, score, text);
    }
    // npoints is POINTS, NOT DOUBLES: `xy` must hold 2*npoints doubles.
    bool polyline(const double* xy, NPoints npoints, const char* label,
                  int64_t id = -1, double score = 0.0, const char* text = NULL) {
        return path_(KIND_POLYLINE, Coords(xy, npoints), npoints.n,
                     label, id, score, text);
    }

    // -- skeletons -------------------------------------------------------------
    //
    // `links` is 2*nlinks indices into THIS shape's points -- or an int[M][2]
    // topology table, which counts as M links. Out-of-range pairs are dropped
    // here rather than shipped for a consumer to trip over.
    bool skeleton(Coords xy, LinkList links, const char* label,
                  int64_t id = -1, double score = 0.0, const char* text = NULL) {
        return skel_(xy, xy.points(), links, links.count(),
                     label, id, score, text);
    }
    // A skeleton with no edges: just the joints.
    bool skeleton(Coords xy, const char* label,
                  int64_t id = -1, double score = 0.0, const char* text = NULL) {
        return skel_(xy, xy.points(), LinkList(), 0u, label, id, score, text);
    }
    // npoints is POINTS and nlinks is LINKS; both are CLAMPED to what the
    // containers actually hold.
    bool skeleton(Coords xy, size_t npoints, LinkList links, size_t nlinks,
                  const char* label, int64_t id = -1, double score = 0.0,
                  const char* text = NULL) {
        return skel_(xy, clamp_points_(npoints, xy),
                     links, (nlinks < links.count() ? nlinks : links.count()),
                     label, id, score, text);
    }
    // Bare pointers: npoints is POINTS (2*npoints doubles) and nlinks is LINKS
    // (2*nlinks ints). Neither can be checked here, hence the explicit units.
    bool skeleton(const double* xy, NPoints npoints, const int* links,
                  NLinks nlinks, const char* label, int64_t id = -1,
                  double score = 0.0, const char* text = NULL) {
        return skel_(Coords(xy, npoints), npoints.n,
                     LinkList(links, nlinks), nlinks.n,
                     label, id, score, text);
    }

    // The compaction path: pass a Skeleton and the renumbering is done for you.
    bool skeleton(const Skeleton& sk, const char* label, int64_t id = -1,
                  double score = 0.0, const char* text = NULL) {
        std::vector<float>   pts;
        std::vector<int32_t> links;
        sk.compact(pts, links);
        if (pts.empty()) return false;
        std::vector<double> xy(pts.begin(), pts.end());
        std::vector<int>    li(links.begin(), links.end());
        return skeleton(Coords(xy), LinkList(li), label, id, score, text);
    }

    // Free-form scalars. A consumer displays them and never interprets them,
    // which is what keeps them usable by a pipeline it has never seen.
    bool field_d(const char* key, double v) {
        if (m_fields.size() >= (size_t) MAX_FIELDS || !std::isfinite(v)) return false;
        Field f; std::memset(&f, 0, sizeof(f));
        f.type = FIELD_D; f.key_off = intern(key); f.dval = v;
        m_fields.push_back(f);
        return true;
    }
    bool field_i(const char* key, long long v) {
        if (m_fields.size() >= (size_t) MAX_FIELDS) return false;
        Field f; std::memset(&f, 0, sizeof(f));
        f.type = FIELD_I; f.key_off = intern(key); f.ival = (int64_t) v;
        m_fields.push_back(f);
        return true;
    }
    bool field_s(const char* key, const char* v) {
        if (m_fields.size() >= (size_t) MAX_FIELDS) return false;
        Field f; std::memset(&f, 0, sizeof(f));
        f.type = FIELD_S; f.key_off = intern(key); f.sval_off = intern(v);
        m_fields.push_back(f);
        return true;
    }

    bool   empty()       const { return m_shapes.empty() && m_fields.empty(); }
    size_t shape_count() const { return m_shapes.size(); }

    // The serialised blob, g_malloc'd; the caller owns it. publish() uses this;
    // it is exposed so a test can build a payload without a batch meta.
    void* build(uint32_t* out_bytes) const {
        const uint32_t hb = (uint32_t) sizeof(Header);
        const uint32_t sb = (uint32_t) (m_shapes.size() * sizeof(Shape));
        const uint32_t pb = (uint32_t) (m_points.size() * sizeof(Point));
        const uint32_t lb = (uint32_t) (m_links.size()  * sizeof(Link));
        const uint32_t fb = (uint32_t) (m_fields.size() * sizeof(Field));
        const uint32_t tb = (uint32_t) m_strings.size();
        const uint32_t total = hb + sb + pb + lb + fb + tb;

        guint8* blob = (guint8*) g_try_malloc0((gsize) total);
        if (!blob) { if (out_bytes) *out_bytes = 0; return NULL; }
        Header* h = (Header*) blob;
        h->magic       = (uint32_t) MAGIC;
        h->version     = (uint32_t) VERSION;
        h->total_bytes = total;
        h->width       = m_w;
        h->height      = m_h;
        h->producer    = m_producer;
        h->n_shapes = (uint32_t) m_shapes.size();  h->off_shapes = hb;
        h->n_points = (uint32_t) m_points.size();  h->off_points = hb + sb;
        h->n_links  = (uint32_t) m_links.size();   h->off_links  = hb + sb + pb;
        h->n_fields = (uint32_t) m_fields.size();  h->off_fields = hb + sb + pb + lb;
        h->off_strings   = hb + sb + pb + lb + fb;
        h->strings_bytes = tb;

        if (sb) std::memcpy(blob + h->off_shapes,  &m_shapes[0],  sb);
        if (pb) std::memcpy(blob + h->off_points,  &m_points[0],  pb);
        if (lb) std::memcpy(blob + h->off_links,   &m_links[0],   lb);
        if (fb) std::memcpy(blob + h->off_fields,  &m_fields[0],  fb);
        if (tb) std::memcpy(blob + h->off_strings, &m_strings[0], tb);

        if (out_bytes) *out_bytes = total;
        return blob;
    }

    // Attach to the frame. `frame_meta` is an NvDsFrameMeta* taken as void*,
    // which the nvcustomprocess struct passes straight through.
    //
    // Returns false and allocates NOTHING when there is nothing to publish,
    // which is what makes the sinks omit their keys entirely rather than emit
    // an empty array.
    bool publish(void* frame_meta) {
        NvDsFrameMeta* fm = (NvDsFrameMeta*) frame_meta;
        NvDsUserMeta* um = acquire_(fm ? fm->base_meta.batch_meta : NULL);
        if (!um) return false;
        nvds_add_user_meta_to_frame(fm, um);
        return true;
    }

    // The same payload, attached to one object instead of the frame.
    bool publish_to_object(NvDsObjectMeta* ob) {
        NvDsUserMeta* um = acquire_(ob ? ob->base_meta.batch_meta : NULL);
        if (!um) return false;
        nvds_add_user_meta_to_obj(ob, um);
        return true;
    }

private:
    NvDsUserMeta* acquire_(NvDsBatchMeta* bm) {
        if (empty() || !bm) return NULL;
        detail::pin_publisher_library();
        NvDsUserMeta* um = nvds_acquire_user_meta_from_pool(bm);
        if (!um) return NULL;
        void* blob = build(NULL);
        if (!blob) return NULL;             // the pool slot is reclaimed at
                                            // teardown; nothing is half-attached
        um->user_meta_data         = blob;
        um->base_meta.meta_type    = meta_type();
        um->base_meta.copy_func    = (NvDsMetaCopyFunc) detail::copy_shapes_meta;
        um->base_meta.release_func = (NvDsMetaReleaseFunc) detail::release_shapes_meta;
        return um;
    }

    uint32_t intern(const char* s) {
        if (!s || !*s) return 0u;
        std::vector<char> clean;
        detail::sanitize(s, clean);
        if (clean.empty()) return 0u;
        if (m_strings.size() + clean.size() + 1u > (size_t) MAX_POOL) return 0u;
        const uint32_t off = (uint32_t) m_strings.size();
        m_strings.insert(m_strings.end(), clean.begin(), clean.end());
        m_strings.push_back('\0');
        return off;
    }

    void init_(Shape& s, uint32_t kind, const char* label, int64_t id,
               double score, const char* text) {
        std::memset(&s, 0, sizeof(s));
        s.kind      = kind;
        s.id        = (id < 0) ? -1 : id;
        s.score     = std::isfinite(score) ? (float) score : 0.0f;
        s.label_off = intern(label);
        s.text_off  = intern(text);
    }

    // A supplied count is never trusted above the container it came with. This
    // is what turns the classic `xy.size()` mistake into the right answer
    // instead of a read off the end.
    static size_t clamp_points_(size_t npoints, const Coords& xy) {
        return npoints < xy.points() ? npoints : xy.points();
    }

    bool skel_(const Coords& xy, size_t npoints, const LinkList& links,
               size_t nlinks, const char* label, int64_t id, double score,
               const char* text) {
        if (!path_(KIND_SKELETON, xy, npoints, label, id, score, text))
            return false;
        add_links_(links, nlinks, npoints);
        return true;
    }

    // All-or-nothing: every coordinate is checked before the first store, so a
    // rejected shape leaves the payload exactly as it was.
    //
    // `npoints` is POINTS, and it is re-clamped HERE against the view as well
    // as at the public overloads -- every path into this function has already
    // done it, so this is the belt-and-braces copy that keeps a future overload
    // from reintroducing the overflow.
    bool path_(uint32_t kind, const Coords& xy, size_t npoints,
               const char* label, int64_t id, double score, const char* text) {
        if (xy.empty() || npoints == 0) return false;
        if (npoints > xy.points()) npoints = xy.points();
        if (m_shapes.size() >= (size_t) MAX_SHAPES) return false;
        if (npoints > (size_t) MAX_POINTS) return false;
        if (npoints > (size_t) MAX_POINTS - m_points.size()) return false;
        for (size_t i = 0; i < npoints; ++i)
            if (!detail::finite_coord(xy.x(i)) ||
                !detail::finite_coord(xy.y(i))) return false;

        Shape s;
        init_(s, kind, label, id, score, text);
        s.first_point = (uint32_t) m_points.size();
        for (size_t i = 0; i < npoints; ++i) {
            Point p;
            p.x = (float) xy.x(i);
            p.y = (float) xy.y(i);
            m_points.push_back(p);
        }
        s.n_points   = (uint32_t) npoints;
        s.first_link = (uint32_t) m_links.size();
        m_shapes.push_back(s);
        return true;
    }

    // `nlinks` is LINKS, clamped against the link view for the same reason
    // path_ re-clamps npoints: a count larger than the caller's array must cost
    // the extra limbs, never a read past the end.
    void add_links_(const LinkList& links, size_t nlinks, size_t npoints) {
        Shape& s = m_shapes.back();
        if (nlinks > links.count()) nlinks = links.count();
        if (nlinks > (size_t) MAX_LINKS) nlinks = (size_t) MAX_LINKS;
        for (size_t i = 0; i < nlinks; ++i) {
            if (m_links.size() >= (size_t) MAX_LINKS) break;
            const int a = links.a(i), b = links.b(i);
            // a == b is a degenerate self-edge: legal JSON, but it draws a
            // zero-length limb and some renderers mishandle it. Rejected on
            // this path too, matching Skeleton::link().
            if (a < 0 || b < 0 || a == b ||
                (size_t) a >= npoints || (size_t) b >= npoints) continue;
            Link L; L.a = (int32_t) a; L.b = (int32_t) b;
            m_links.push_back(L);
        }
        s.n_links = (uint32_t) (m_links.size() - s.first_link);
    }

    uint32_t m_w, m_h, m_producer;
    std::vector<Shape> m_shapes;
    std::vector<Point> m_points;
    std::vector<Link>  m_links;
    std::vector<Field> m_fields;
    std::vector<char>  m_strings;
};

// -- forwarding onto the VMS overlay contract --------------------------------
//
// Templated on the line type so THIS header never has to include shape_json.h,
// which the JSON and MQTT sinks do not ship and whose include order would
// otherwise become load-bearing in vms_detect.cpp. A translation unit that
// never calls it emits nothing:
//
//     std::vector<render_meta::View> vs;
//     render_meta::views_on_frame(fm, vs);
//     for (size_t i = 0; i < vs.size(); ++i)
//         render_meta::to_shape_line(vs[i], line, ox, oy);
//
// ox/oy shift a per-source geometry into that source's tile on the composed
// canvas, exactly as vms_detect's existing box loop already does; sx/sy rescale
// when the console's canvas is not the space the renderer published in.
//
// `max_shapes` is a liveness guard, not a formatting one: event_sink drops a
// console that cannot drain and treats a partial send as failure, so one
// runaway frame can knock consoles off the wall. Returns the number of shapes
// written.
//
// NOTE shape_json::path()/skeleton() hard-code id = -1, so a tracked pose
// cannot carry its track id to the console until those grow an id parameter.
// The id IS in this payload and IS emitted by the JSON and MQTT sinks.
template <class ShapeLine>
inline size_t to_shape_line(const View& v, ShapeLine& line,
                            double ox = 0.0, double oy = 0.0,
                            double sx = 1.0, double sy = 1.0,
                            size_t max_shapes = (size_t) MAX_SHAPES) {
    if (!v.valid()) return 0;
    std::vector<double> xy;
    std::vector<int>    idx;
    size_t written = 0;
    for (uint32_t i = 0; i < v.n_shapes() && written < max_shapes; ++i) {
        const Shape& s = v.shape(i);
        const char* label = v.str(s.label_off);
        const char* text  = v.str(s.text_off);
        const char* txt   = *text ? text : NULL;

        if (s.kind == (uint32_t) KIND_BOX) {
            line.box((double) s.x * sx + ox, (double) s.y * sy + oy,
                     (double) s.w * sx, (double) s.h * sy,
                     label, s.id, (double) s.score, txt);
            ++written;
            continue;
        }

        const Point* pts = NULL;
        const uint32_t np = v.points_of(s, &pts);
        if (np == 0) continue;
        xy.clear();
        xy.reserve((size_t) np * 2u);
        for (uint32_t p = 0; p < np; ++p) {
            xy.push_back((double) pts[p].x * sx + ox);
            xy.push_back((double) pts[p].y * sy + oy);
        }

        if (s.kind == (uint32_t) KIND_SKELETON) {
            const Link* lk = NULL;
            const uint32_t nl = v.links_of(s, &lk);
            idx.clear();
            idx.reserve((size_t) nl * 2u);
            for (uint32_t k = 0; k < nl; ++k) {
                idx.push_back((int) lk[k].a);
                idx.push_back((int) lk[k].b);
            }
            line.skeleton(&xy[0], np,
                          idx.empty() ? (const int*) NULL : &idx[0], nl,
                          label, (double) s.score);
        } else if (s.kind == (uint32_t) KIND_POLYGON) {
            line.polygon(&xy[0], np, label, txt);
        } else {
            line.polyline(&xy[0], np, label, txt);
        }
        ++written;
    }
    for (uint32_t i = 0; i < v.n_fields(); ++i) {
        const Field& f = v.field(i);
        const char* key = v.str(f.key_off);
        if (!*key) continue;
        if (f.type == (uint32_t) FIELD_I)      line.field_i(key, (long long) f.ival);
        else if (f.type == (uint32_t) FIELD_S) line.field_s(key, v.str(f.sval_off));
        else                                   line.field_d(key, f.dval);
    }
    return written;
}

}  // namespace render_meta

#endif  // RENDER_META_H_
