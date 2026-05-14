/*
 * FaceX implementation of the NIST FRVT 1:1 (FRTE Verification) API.
 *
 * Loads three ONNX models from configDir:
 *   facex_detect.onnx     — face detector (FCOS-style, 320x320 RGB normalized)
 *   facex_landmark.onnx   — 98-point WFLW landmarks (112x112 RGB)
 *   facex_xs.onnx         — 512-dim recognition embedding (112x112 RGB, L2 normalized)
 *
 * Each template is a fixed-size 512 × 4 = 2048-byte L2-normalized float32 vector.
 * Match = cosine similarity rescaled to [0, 1].
 *
 * Build for Linux x86_64 (NIST evaluation platform):
 *   cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
 *   cmake --build build --config Release
 *   -> libfrvt_11_facex_001.so
 */

#include "frvt11.h"
#include "frvt_structs.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace FRVT;

namespace {

// ===== ArcFace 5-point alignment template (InsightFace, 112×112 output) =====
constexpr float ARC_TEMPLATE[5][2] = {
    {38.2946f, 51.6963f},   // left eye center
    {73.5318f, 51.5014f},   // right eye center
    {56.0252f, 71.7366f},   // nose tip
    {41.5493f, 92.3655f},   // left mouth corner
    {70.7299f, 92.2041f},   // right mouth corner
};

// WFLW 98-point indices used to derive the 5 alignment keypoints.
constexpr int WFLW_LEFT_EYE_PTS[]  = {60, 61, 62, 63, 64, 65, 66, 67};
constexpr int WFLW_RIGHT_EYE_PTS[] = {68, 69, 70, 71, 72, 73, 74, 75};
constexpr int WFLW_NOSE_TIP        = 54;
constexpr int WFLW_LEFT_MOUTH      = 76;
constexpr int WFLW_RIGHT_MOUTH     = 82;

constexpr int DET_SIZE = 320;
constexpr int LM_SIZE  = 112;
constexpr int REC_SIZE = 112;
constexpr int EMB_DIM  = 512;

// ===== Similarity transform (rotation + uniform scale + translation) =====
// Umeyama algorithm minimizing sum of squared errors between two point sets.
// Inputs: src[5][2] (detected) and dst[5][2] (template). Output: 2x3 affine M.
static void umeyama_5pt(const float src[5][2], const float dst[5][2], float M[6])
{
    float src_cx = 0, src_cy = 0, dst_cx = 0, dst_cy = 0;
    for (int i = 0; i < 5; i++) {
        src_cx += src[i][0]; src_cy += src[i][1];
        dst_cx += dst[i][0]; dst_cy += dst[i][1];
    }
    src_cx /= 5; src_cy /= 5; dst_cx /= 5; dst_cy /= 5;
    float a = 0, b = 0, c = 0;
    for (int i = 0; i < 5; i++) {
        float sx = src[i][0] - src_cx, sy = src[i][1] - src_cy;
        float dx = dst[i][0] - dst_cx, dy = dst[i][1] - dst_cy;
        a += sx*dx + sy*dy;
        b += sx*dy - sy*dx;
        c += sx*sx + sy*sy;
    }
    float scale = std::hypot(a, b) / std::max(c, 1e-6f);
    float cos_t = a / std::hypot(a, b);
    float sin_t = b / std::hypot(a, b);
    M[0] = scale * cos_t;   M[1] = -scale * sin_t;
    M[3] = scale * sin_t;   M[4] =  scale * cos_t;
    M[2] = dst_cx - M[0] * src_cx - M[1] * src_cy;
    M[5] = dst_cy - M[3] * src_cx - M[4] * src_cy;
}

// Warp 24-bit RGB image into a dst×dst RGB plane using inverse affine + bilinear.
// We compute the inverse of the 2x3 forward map, then for each dst pixel sample src.
static void warp_affine_bilinear(
    const uint8_t* src, int sw, int sh, int sd,
    uint8_t* dst, int dst_size, const float fwd[6])
{
    float det = fwd[0] * fwd[4] - fwd[1] * fwd[3];
    if (std::fabs(det) < 1e-9f) det = 1e-9f;
    float inv[6];
    inv[0] =  fwd[4] / det; inv[1] = -fwd[1] / det;
    inv[3] = -fwd[3] / det; inv[4] =  fwd[0] / det;
    inv[2] = -(inv[0] * fwd[2] + inv[1] * fwd[5]);
    inv[5] = -(inv[3] * fwd[2] + inv[4] * fwd[5]);

    int channels = (sd == 24) ? 3 : 1;
    for (int y = 0; y < dst_size; y++) {
        for (int x = 0; x < dst_size; x++) {
            float fx = inv[0] * x + inv[1] * y + inv[2];
            float fy = inv[3] * x + inv[4] * y + inv[5];
            int x0 = static_cast<int>(std::floor(fx));
            int y0 = static_cast<int>(std::floor(fy));
            int x1 = x0 + 1, y1 = y0 + 1;
            float wx = fx - x0, wy = fy - y0;
            uint8_t* dp = dst + (y * dst_size + x) * 3;
            if (x0 < 0 || x1 >= sw || y0 < 0 || y1 >= sh) {
                dp[0] = dp[1] = dp[2] = 0;
                continue;
            }
            const uint8_t* p00 = src + (y0 * sw + x0) * channels;
            const uint8_t* p01 = src + (y0 * sw + x1) * channels;
            const uint8_t* p10 = src + (y1 * sw + x0) * channels;
            const uint8_t* p11 = src + (y1 * sw + x1) * channels;
            for (int c = 0; c < 3; c++) {
                int sc = (channels == 3) ? c : 0;
                float v = (1 - wx) * (1 - wy) * p00[sc]
                        + wx       * (1 - wy) * p01[sc]
                        + (1 - wx) * wy       * p10[sc]
                        + wx       * wy       * p11[sc];
                dp[c] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, v + 0.5f)));
            }
        }
    }
}

// Convert HWC RGB uint8 → CHW float32 normalized (val - 127.5) / 128.
static void hwc_to_chw_norm(const uint8_t* src, int size, float* dst)
{
    int n = size * size;
    for (int i = 0; i < n; i++) {
        dst[i]         = (src[i * 3 + 0] - 127.5f) / 128.0f;
        dst[n + i]     = (src[i * 3 + 1] - 127.5f) / 128.0f;
        dst[2 * n + i] = (src[i * 3 + 2] - 127.5f) / 128.0f;
    }
}

// Letterbox into DET_SIZE × DET_SIZE (top-left paste, scale by min ratio).
static void letterbox_det(const uint8_t* src, int sw, int sh, int sd,
                          uint8_t* dst, float& outScale)
{
    float scale = static_cast<float>(DET_SIZE) / std::max(sw, sh);
    outScale = scale;
    int new_w = std::min(DET_SIZE, static_cast<int>(sw * scale));
    int new_h = std::min(DET_SIZE, static_cast<int>(sh * scale));
    std::memset(dst, 0, DET_SIZE * DET_SIZE * 3);
    int channels = (sd == 24) ? 3 : 1;
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            float fx = x / scale, fy = y / scale;
            int x0 = static_cast<int>(fx), y0 = static_cast<int>(fy);
            if (x0 >= sw) x0 = sw - 1;
            if (y0 >= sh) y0 = sh - 1;
            const uint8_t* p = src + (y0 * sw + x0) * channels;
            uint8_t* dp = dst + (y * DET_SIZE + x) * 3;
            if (channels == 3) {
                dp[0] = p[0]; dp[1] = p[1]; dp[2] = p[2];
            } else {
                dp[0] = dp[1] = dp[2] = p[0];
            }
        }
    }
}

// ===== FCOS decoder for our facex_detect.onnx outputs =====
struct Detection { float x1, y1, x2, y2, score; };

static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static std::vector<Detection> decode_detector(
    const std::vector<Ort::Value>& outs,
    const std::vector<const char*>& names,
    float scoreThr = 0.40f)
{
    constexpr int STRIDES[3] = {8, 16, 32};
    std::vector<Detection> dets;
    for (int li = 0; li < 3; li++) {
        int s = STRIDES[li];
        // Outputs are ordered: cls_p3, box_p3, kps_p3, cls_p4, ...
        const float* cls = outs[li * 3 + 0].GetTensorData<float>();
        const float* box = outs[li * 3 + 1].GetTensorData<float>();
        int H = DET_SIZE / s, W = DET_SIZE / s, N = H * W;
        for (int i = 0; i < N; i++) {
            float sc = sigmoid(cls[i]);
            if (sc < scoreThr) continue;
            int y = i / W, x = i % W;
            float cx = (x + 0.5f) * s, cy = (y + 0.5f) * s;
            float l = std::max(0.0f, box[0 * N + i]) * s;
            float t = std::max(0.0f, box[1 * N + i]) * s;
            float r = std::max(0.0f, box[2 * N + i]) * s;
            float b = std::max(0.0f, box[3 * N + i]) * s;
            Detection d{cx - l, cy - t, cx + r, cy + b, sc};
            if (d.x2 <= d.x1 || d.y2 <= d.y1) continue;
            float w = d.x2 - d.x1, h = d.y2 - d.y1;
            float ar = w / std::max(1.0f, h);
            if (ar < 0.5f || ar > 1.75f) continue;
            dets.push_back(d);
        }
    }
    // NMS
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    std::vector<bool> sup(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (sup[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (sup[j]) continue;
            float ix1 = std::max(dets[i].x1, dets[j].x1);
            float iy1 = std::max(dets[i].y1, dets[j].y1);
            float ix2 = std::min(dets[i].x2, dets[j].x2);
            float iy2 = std::min(dets[i].y2, dets[j].y2);
            float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
            float inter = iw * ih;
            float ai = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
            float aj = (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1);
            if (inter / (ai + aj - inter + 1e-7f) > 0.4f) sup[j] = true;
        }
    }
    return kept;
}

// ===== Engine =====
class FaceXEngine {
public:
    bool init(const std::string& configDir)
    {
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "facex_frvt");
            opts_.SetIntraOpNumThreads(1);   // NIST forks for parallelism
            opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            det_ = std::make_unique<Ort::Session>(*env_, (configDir + "/facex_detect.onnx").c_str(), opts_);
            lm_  = std::make_unique<Ort::Session>(*env_, (configDir + "/facex_landmark.onnx").c_str(), opts_);
            rec_ = std::make_unique<Ort::Session>(*env_, (configDir + "/facex_xs.onnx").c_str(), opts_);
            return true;
        } catch (const std::exception&) { return false; }
    }

    // Returns the best face's 5 keypoints (left_eye, right_eye, nose, l_mouth, r_mouth)
    // in original-image pixel coords, plus the bbox.
    bool detect_align(const Image& img,
                      float kps[5][2], float bbox[4]) const
    {
        // 1) Detect at 320×320 letterbox
        std::vector<uint8_t> letter(DET_SIZE * DET_SIZE * 3);
        float scale; letterbox_det(img.data.get(), img.width, img.height, img.depth,
                                    letter.data(), scale);
        std::vector<float> detInput(3 * DET_SIZE * DET_SIZE);
        hwc_to_chw_norm(letter.data(), DET_SIZE, detInput.data());

        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> inShape{1, 3, DET_SIZE, DET_SIZE};
        Ort::Value inTensor = Ort::Value::CreateTensor<float>(
            mem, detInput.data(), detInput.size(), inShape.data(), inShape.size());
        const char* inName = "input";
        const std::vector<const char*> outNames = {
            "cls_p3", "box_p3", "kps_p3",
            "cls_p4", "box_p4", "kps_p4",
            "cls_p5", "box_p5", "kps_p5",
        };
        auto outs = det_->Run(Ort::RunOptions{nullptr}, &inName, &inTensor, 1,
                               outNames.data(), outNames.size());
        auto dets = decode_detector(outs, outNames);
        if (dets.empty()) return false;

        // Pick largest face
        Detection best = *std::max_element(dets.begin(), dets.end(),
            [](const Detection& a, const Detection& b) {
                return (a.x2 - a.x1) * (a.y2 - a.y1) < (b.x2 - b.x1) * (b.y2 - b.y1);
            });
        // Un-letterbox bbox back to original image
        bbox[0] = best.x1 / scale; bbox[1] = best.y1 / scale;
        bbox[2] = best.x2 / scale; bbox[3] = best.y2 / scale;
        bbox[0] = std::max(0.0f, std::min(static_cast<float>(img.width - 1), bbox[0]));
        bbox[2] = std::max(0.0f, std::min(static_cast<float>(img.width - 1), bbox[2]));
        bbox[1] = std::max(0.0f, std::min(static_cast<float>(img.height - 1), bbox[1]));
        bbox[3] = std::max(0.0f, std::min(static_cast<float>(img.height - 1), bbox[3]));

        // 2) Crop+resize face to 112×112 for landmark net
        float fw = bbox[2] - bbox[0], fh = bbox[3] - bbox[1];
        float side = std::max(fw, fh) * 1.3f;
        float cx = (bbox[0] + bbox[2]) / 2.0f, cy = (bbox[1] + bbox[3]) / 2.0f;
        float sx = cx - side / 2.0f, sy = cy - side / 2.0f;
        std::vector<uint8_t> faceCrop(LM_SIZE * LM_SIZE * 3);
        int channels = (img.depth == 24) ? 3 : 1;
        for (int y = 0; y < LM_SIZE; y++) {
            for (int x = 0; x < LM_SIZE; x++) {
                float fx_ = sx + (x + 0.5f) * side / LM_SIZE;
                float fy_ = sy + (y + 0.5f) * side / LM_SIZE;
                int xi = std::max(0, std::min(img.width - 1, static_cast<int>(fx_)));
                int yi = std::max(0, std::min(img.height - 1, static_cast<int>(fy_)));
                const uint8_t* p = img.data.get() + (yi * img.width + xi) * channels;
                uint8_t* dp = faceCrop.data() + (y * LM_SIZE + x) * 3;
                if (channels == 3) { dp[0] = p[0]; dp[1] = p[1]; dp[2] = p[2]; }
                else { dp[0] = dp[1] = dp[2] = p[0]; }
            }
        }
        std::vector<float> lmInput(3 * LM_SIZE * LM_SIZE);
        hwc_to_chw_norm(faceCrop.data(), LM_SIZE, lmInput.data());
        std::array<int64_t, 4> lmShape{1, 3, LM_SIZE, LM_SIZE};
        Ort::Value lmTensor = Ort::Value::CreateTensor<float>(
            mem, lmInput.data(), lmInput.size(), lmShape.data(), lmShape.size());
        const char* lmInName = "input";
        const char* lmOutName = "landmarks";   // model exports a single (1, 196) tensor
        auto lmOuts = lm_->Run(Ort::RunOptions{nullptr}, &lmInName, &lmTensor, 1,
                                &lmOutName, 1);
        const float* lmData = lmOuts[0].GetTensorData<float>();
        // 98 (x, y) in [0, 1] of the cropped 112×112 → map back to original
        auto cropToOrig = [&](float xc, float yc, float out[2]) {
            out[0] = sx + xc * side;
            out[1] = sy + yc * side;
        };
        auto getPt = [&](int idx, float p[2]) { cropToOrig(lmData[idx*2], lmData[idx*2+1], p); };

        // Average eye centers from 8 contour points each
        kps[0][0] = kps[0][1] = 0;
        for (int i : WFLW_LEFT_EYE_PTS) { float p[2]; getPt(i, p); kps[0][0] += p[0]; kps[0][1] += p[1]; }
        kps[0][0] /= 8; kps[0][1] /= 8;
        kps[1][0] = kps[1][1] = 0;
        for (int i : WFLW_RIGHT_EYE_PTS) { float p[2]; getPt(i, p); kps[1][0] += p[0]; kps[1][1] += p[1]; }
        kps[1][0] /= 8; kps[1][1] /= 8;
        getPt(WFLW_NOSE_TIP,    kps[2]);
        getPt(WFLW_LEFT_MOUTH,  kps[3]);
        getPt(WFLW_RIGHT_MOUTH, kps[4]);
        return true;
    }

    // Align the face using 5-pt similarity transform, then run recognition.
    bool extract_embedding(const Image& img, const float kps[5][2], float emb[EMB_DIM]) const
    {
        float M[6];
        umeyama_5pt(kps, ARC_TEMPLATE, M);
        std::vector<uint8_t> aligned(REC_SIZE * REC_SIZE * 3);
        warp_affine_bilinear(img.data.get(), img.width, img.height, img.depth,
                              aligned.data(), REC_SIZE, M);
        std::vector<float> recInput(3 * REC_SIZE * REC_SIZE);
        hwc_to_chw_norm(aligned.data(), REC_SIZE, recInput.data());

        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 4> shape{1, 3, REC_SIZE, REC_SIZE};
        Ort::Value t = Ort::Value::CreateTensor<float>(
            mem, recInput.data(), recInput.size(), shape.data(), shape.size());
        const char* in = "input"; const char* out = "embedding";
        auto outs = rec_->Run(Ort::RunOptions{nullptr}, &in, &t, 1, &out, 1);
        const float* e = outs[0].GetTensorData<float>();
        // re-normalize defensively
        float norm = 0;
        for (int i = 0; i < EMB_DIM; i++) norm += e[i] * e[i];
        norm = std::sqrt(std::max(norm, 1e-12f));
        for (int i = 0; i < EMB_DIM; i++) emb[i] = e[i] / norm;
        return true;
    }

private:
    std::unique_ptr<Ort::Env> env_;
    Ort::SessionOptions opts_;
    std::unique_ptr<Ort::Session> det_;
    std::unique_ptr<Ort::Session> lm_;
    std::unique_ptr<Ort::Session> rec_;
};

// ===== Template format =====
// Magic "FXT1" + uint8 valid_flag + 512 × float32 L2-normalized embedding.
// 4 + 1 + 2048 = 2053 bytes per template.
constexpr uint32_t TEMPL_MAGIC = 0x31545846;   // "FXT1" little-endian
constexpr size_t TEMPL_SIZE = sizeof(uint32_t) + 1 + EMB_DIM * sizeof(float);

static void pack_template(const float* emb, bool valid, std::vector<uint8_t>& out)
{
    out.resize(TEMPL_SIZE);
    std::memcpy(out.data(), &TEMPL_MAGIC, 4);
    out[4] = valid ? 1 : 0;
    std::memcpy(out.data() + 5, emb, EMB_DIM * sizeof(float));
}

static bool unpack_template(const std::vector<uint8_t>& t, bool& valid, const float*& emb)
{
    if (t.size() != TEMPL_SIZE) return false;
    uint32_t m; std::memcpy(&m, t.data(), 4);
    if (m != TEMPL_MAGIC) return false;
    valid = (t[4] == 1);
    emb = reinterpret_cast<const float*>(t.data() + 5);
    return true;
}

// ===== Concrete Interface implementation =====
class FaceXFRVT : public FRVT_11::Interface {
public:
    ReturnStatus initialize(const std::string& configDir) override
    {
        if (engine_.init(configDir)) return ReturnStatus(ReturnCode::Success);
        return ReturnStatus(ReturnCode::ConfigError, "failed to load ONNX models");
    }

    ReturnStatus createFaceTemplate(
        const std::vector<Image>& faces,
        TemplateRole /*role*/,
        std::vector<uint8_t>& templ,
        std::vector<EyePair>& eyeCoordinates) override
    {
        if (faces.empty()) {
            float zero[EMB_DIM] = {0};
            pack_template(zero, false, templ);
            eyeCoordinates.clear();
            return ReturnStatus(ReturnCode::Success);
        }
        // Average embeddings from up to 5 images of the same person.
        std::vector<float> sum(EMB_DIM, 0.0f);
        int valid_n = 0;
        eyeCoordinates.clear(); eyeCoordinates.reserve(faces.size());
        for (size_t i = 0; i < faces.size() && i < 5; i++) {
            float kps[5][2], bbox[4], emb[EMB_DIM];
            if (engine_.detect_align(faces[i], kps, bbox) &&
                engine_.extract_embedding(faces[i], kps, emb)) {
                for (int j = 0; j < EMB_DIM; j++) sum[j] += emb[j];
                valid_n++;
                eyeCoordinates.emplace_back(
                    true, true,
                    static_cast<uint16_t>(kps[0][0]),
                    static_cast<uint16_t>(kps[0][1]),
                    static_cast<uint16_t>(kps[1][0]),
                    static_cast<uint16_t>(kps[1][1]));
            } else {
                eyeCoordinates.emplace_back();
            }
        }
        // L2-normalize the sum
        float norm = 0;
        for (int j = 0; j < EMB_DIM; j++) norm += sum[j] * sum[j];
        norm = std::sqrt(std::max(norm, 1e-12f));
        for (int j = 0; j < EMB_DIM; j++) sum[j] /= norm;
        pack_template(sum.data(), valid_n > 0, templ);
        return ReturnStatus(ReturnCode::Success);
    }

    ReturnStatus createIrisTemplate(
        const std::vector<Image>&, TemplateRole,
        std::vector<uint8_t>& templ, std::vector<IrisAnnulus>&) override
    {
        templ.clear();
        return ReturnStatus(ReturnCode::NotImplemented, "iris not supported");
    }

    ReturnStatus createFaceTemplate(
        const Image& image, TemplateRole /*role*/,
        std::vector<std::vector<uint8_t>>& templs,
        std::vector<EyePair>& eyeCoordinates) override
    {
        // Single-image multi-face path. We currently detect+template the largest face.
        float kps[5][2], bbox[4], emb[EMB_DIM];
        std::vector<uint8_t> t;
        if (engine_.detect_align(image, kps, bbox) &&
            engine_.extract_embedding(image, kps, emb)) {
            pack_template(emb, true, t);
            templs.push_back(t);
            eyeCoordinates.emplace_back(
                true, true,
                static_cast<uint16_t>(kps[0][0]),
                static_cast<uint16_t>(kps[0][1]),
                static_cast<uint16_t>(kps[1][0]),
                static_cast<uint16_t>(kps[1][1]));
        } else {
            float zero[EMB_DIM] = {0};
            pack_template(zero, false, t);
            templs.push_back(t);
            eyeCoordinates.emplace_back();
        }
        return ReturnStatus(ReturnCode::Success);
    }

    ReturnStatus matchTemplates(
        const std::vector<uint8_t>& verif,
        const std::vector<uint8_t>& enroll,
        double& score) override
    {
        bool va, ea; const float* e1; const float* e2;
        if (!unpack_template(verif, va, e1) || !unpack_template(enroll, ea, e2)) {
            score = -1;
            return ReturnStatus(ReturnCode::TemplateFormatError);
        }
        if (!va || !ea) {
            score = -1;
            return ReturnStatus(ReturnCode::VerifTemplateError);
        }
        double dot = 0;
        for (int i = 0; i < EMB_DIM; i++) dot += static_cast<double>(e1[i]) * e2[i];
        // Map cosine [-1, 1] → [0, 1] then to [0, 1000] for higher-resolution score
        score = (dot + 1.0) * 500.0;
        return ReturnStatus(ReturnCode::Success);
    }

private:
    FaceXEngine engine_;
};

}  // namespace

namespace FRVT_11 {
std::shared_ptr<Interface> Interface::getImplementation()
{
    return std::make_shared<FaceXFRVT>();
}
}
