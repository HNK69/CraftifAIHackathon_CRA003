// meta_json.cpp -- turn NvDsBatchMeta into JSON and ship it, for the
// `nvcustomprocess` plugin.
//
// Writes one JSON document describing every frame and every detection in it.
//
// This pipeline writes FILES -- an annotated mp4 and this .json. Publishing the
// same metadata live over MQTT is ../mqtt_pipeline, which exists separately
// because a file writer and a message publisher want different things: a file
// wants every frame and a clean document, a bus wants rate limiting, topics,
// reconnection and a receiver. Mixing them made both harder to run.
//
// The element is placed AFTER nvtracker, which is what makes `id` a persistent
// object id rather than a per-frame index. It reads metadata only and never
// touches a pixel.
//
// CONFIGURATION (env vars -- the plugin has no config-file property)
//
//     JSON_OUT         output path; unset = do not write a file
//     JSON_SOURCE      source description, recorded in the file header
//     JSON_MIN_CONF    0.0      drop objects below this detector confidence.
//                               An object whose payload is NOT its bbox -- a
//                               classifier result, an instance mask -- has no
//                               detector confidence and is kept regardless.

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <glib.h>

#include "nvcustomprocess.h"
#include "nvdsmeta.h"
#include "frame_json.h"

namespace {

// --- file sink --------------------------------------------------------------
FILE* g_out = nullptr;
bool  g_first_frame = true;

// --- shared -----------------------------------------------------------------
bool   g_init_done = false;
double g_min_conf = 0.0;
uint64_t g_frames = 0, g_objects = 0;

// The payload shape lives in frame_json.h, shared with ../mqtt_pipeline so a file
// consumer and a bus consumer see identical bytes. See that header for why.
using frame_json::appendf;

// ===========================================================================
// lifecycle
// ===========================================================================
void close_all() {
    if (g_out) {
        // Close whatever nesting is open; an untouched array is still valid.
        fputs("\n  ]\n}\n", g_out);
        fclose(g_out);
        g_out = nullptr;
    }
    if (g_frames)
        fprintf(stderr, "[meta_json] %" PRIu64 " frames, %" PRIu64 " objects\n",
                g_frames, g_objects);
}

// Runs at process exit, which is where the document gets closed. `gst-launch-1.0
// -e` turns SIGINT into EOS and exits normally, so this covers ctrl-c too;
// SIGKILL skips it and leaves the file unclosed.
struct Closer { ~Closer() { close_all(); } };
Closer g_closer;

void file_init() {
    const char* path = getenv("JSON_OUT");
    if (!path || !*path) return;
    g_out = fopen(path, "w");
    if (!g_out) {
        fprintf(stderr, "[meta_json] cannot open %s for writing\n", path);
        return;
    }
    const char* src = getenv("JSON_SOURCE");
    std::string head = "{\n  \"pipeline\": \"video_json_export\",\n  \"source\": ";
    frame_json::escape(head, (src && *src) ? src : "");
    fputs(head.c_str(), g_out);
    fprintf(g_out, ",\n  \"min_confidence\": %.3f,\n  \"frames\": [", g_min_conf);
    fprintf(stderr, "[meta_json] file -> %s\n", path);
}

void lazy_init() {
    if (g_init_done) return;
    g_init_done = true;
    const char* mc = getenv("JSON_MIN_CONF");
    if (mc && *mc) g_min_conf = atof(mc);
    file_init();
    if (!g_out)
        fprintf(stderr, "[meta_json] JSON_OUT not set -- writing no metadata\n");
}

}  // namespace

// ===========================================================================
// Entry point: MetaJson -- after the tracker, metadata only
// ===========================================================================
extern "C" bool MetaJson(NvDsCustomPostProcessFrameInCpu* in)
{
    lazy_init();

    NvDsFrameMeta* fm = (NvDsFrameMeta*) in->frame_meta;
    // Returning false aborts the pipeline. A metadata exporter that cannot
    // deliver must not take the video down with it, so every failure path here
    // lets the frames through.
    if (!fm || !g_out) return true;

    std::vector<NvDsObjectMeta*> objs;
    objs.reserve(in->num_obj_metas);
    for (size_t i = 0; i < in->num_obj_metas; ++i) {
        NvDsObjectMeta* ob = (NvDsObjectMeta*) in->obj_metas[i];
        // JSON_MIN_CONF is a DETECTOR gate, and an object whose payload is not
        // in its bbox has no detector confidence to gate on. nvinfer's
        // classifier path fabricates one whole-frame object per frame and
        // attach_metadata_classifier assigns its class_id, object_id and
        // rect_params but never its confidence, so it reads 0.0f -- the value a
        // released pool slot is zeroed to. Gating that object on confidence
        // alone therefore discards the ENTIRE classification result, and every
        // attribute hanging off it, for any JSON_MIN_CONF above zero: the same
        // 60-frame classifier run emits 60 objects at the 0.0 default and 0 at
        // 0.3. Same for an instance mask, whose answer is in mask_params.
        //
        // frame_json::has_payload() is exactly that test and this is the call
        // site its own comment prescribes. It reads two pointers on an object
        // the gate already rejected, so a detector -- where both are always
        // NULL -- keeps byte-identical output and pays nothing measurable.
        if (ob && (ob->confidence >= g_min_conf || frame_json::has_payload(ob)))
            objs.push_back(ob);
    }

    std::string payload;
    payload.reserve(128 + 160 * objs.size());
    frame_json::frame(payload, fm, in->W, in->H, objs);

    if (g_out) {
        // One frame object per line: a completed run is a single valid JSON
        // document, and a killed run still leaves one valid object per line from
        // "frames": onward, which is recoverable.
        if (!g_first_frame) fputc(',', g_out);
        g_first_frame = false;
        fputs("\n    ", g_out);
        fputs(payload.c_str(), g_out);
        // Flushing every frame costs little at video rates and makes the file
        // readable while the pipeline is still running.
        fflush(g_out);
    }

    g_objects += objs.size();
    ++g_frames;
    return true;
}
