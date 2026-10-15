// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[YOLO11s_det] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // Engine output shape [1, 84, 8400] → after batch strip: numDims=2, d={84, 8400}
    // d[0] = numAttributes = 4 + numClasses (84), d[1] = numProposals (8400)
    // Data layout: data[attrIdx * numProposals + proposalIdx]

    const unsigned numAttrs = layer0.inferDims.d[0];   // 84
    const unsigned numProposals = layer0.inferDims.d[1]; // 8400

    // Sanity: numAttrs should be 4 + numClasses
    if (numAttrs < 4u + numClasses) return false;

    // Collect candidates per class for NMS
    struct RawDet {
        float cx, cy, w, h, score;
        unsigned classId;
    };

    // Group by class for per-class NMS
    std::vector<std::vector<RawDet>> perClassDets(numClasses);

    for (unsigned i = 0; i < numProposals; ++i) {
        // cx, cy, w, h are in network-input pixel coords (letterboxed)
        float cx = data[0u * numProposals + i];
        float cy = data[1u * numProposals + i];
        float bw = data[2u * numProposals + i];
        float bh = data[3u * numProposals + i];

        // Find best class score — scores are post-sigmoid probabilities
        float bestScore = 0.0f;
        unsigned bestClassId = 0;
        for (unsigned c = 0; c < numClasses; ++c) {
            float s = data[(4u + c) * numProposals + i];
            if (s > bestScore) {
                bestScore = s;
                bestClassId = c;
            }
        }

        // Threshold
        float thr = (bestClassId < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClassId]
                     : 0.3f;
        if (bestScore < thr) continue;

        // Only emit classes 0 (person), 1 (bicycle), 2 (car)
        if (bestClassId != 0 && bestClassId != 1 && bestClassId != 2) continue;

        RawDet rd;
        rd.cx = cx;
        rd.cy = cy;
        rd.w = bw;
        rd.h = bh;
        rd.score = bestScore;
        rd.classId = bestClassId;
        perClassDets[bestClassId].push_back(rd);
    }

    // Per-class NMS (IoU threshold 0.45)
    const float nmsIoUThresh = 0.45f;

    for (unsigned c = 0; c < numClasses; ++c) {
        auto& dets = perClassDets[c];
        if (dets.empty()) continue;

        // Sort by score descending
        std::sort(dets.begin(), dets.end(),
                  [](const RawDet& a, const RawDet& b) { return a.score > b.score; });

        std::vector<bool> suppressed(dets.size(), false);

        for (size_t i = 0; i < dets.size(); ++i) {
            if (suppressed[i]) continue;

            float x1_i = dets[i].cx - dets[i].w * 0.5f;
            float y1_i = dets[i].cy - dets[i].h * 0.5f;
            float x2_i = dets[i].cx + dets[i].w * 0.5f;
            float y2_i = dets[i].cy + dets[i].h * 0.5f;
            float area_i = dets[i].w * dets[i].h;

            // Emit this detection (in network-input coords, top-left + size)
            float left   = std::max(x1_i, 0.0f);
            float top    = std::max(y1_i, 0.0f);
            float right  = std::min(x2_i, netW);
            float bottom = std::min(y2_i, netH);
            float bw_out = right - left;
            float bh_out = bottom - top;

            if (bw_out > 1.0f && bh_out > 1.0f) {
                NvDsInferObjectDetectionInfo obj;
                obj.classId = dets[i].classId;
                obj.left = left;
                obj.top = top;
                obj.width = bw_out;
                obj.height = bh_out;
                obj.detectionConfidence = dets[i].score;
                objectList.push_back(obj);
            }

            // Suppress overlapping boxes
            for (size_t j = i + 1; j < dets.size(); ++j) {
                if (suppressed[j]) continue;

                float x1_j = dets[j].cx - dets[j].w * 0.5f;
                float y1_j = dets[j].cy - dets[j].h * 0.5f;
                float x2_j = dets[j].cx + dets[j].w * 0.5f;
                float y2_j = dets[j].cy + dets[j].h * 0.5f;
                float area_j = dets[j].w * dets[j].h;

                float inter_x1 = std::max(x1_i, x1_j);
                float inter_y1 = std::max(y1_i, y1_j);
                float inter_x2 = std::min(x2_i, x2_j);
                float inter_y2 = std::min(y2_i, y2_j);

                float inter_w = std::max(inter_x2 - inter_x1, 0.0f);
                float inter_h = std::max(inter_y2 - inter_y1, 0.0f);
                float inter_area = inter_w * inter_h;

                float unionArea = area_i + area_j - inter_area;
                float iou = (unionArea > 0.0f) ? (inter_area / unionArea) : 0.0f;

                if (iou > nmsIoUThresh) {
                    suppressed[j] = true;
                }
            }
        }
    }
    // ==== END GENERATED DECODE ====
    return true;
}