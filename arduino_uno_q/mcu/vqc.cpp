#include <math.h>

#include "vqc.h"
#include "vqc_params.h"

static const int kDim = 1 << VQC_N_QUBITS;
static float re_[kDim];
static float im_[kDim];
static float czSign_[kDim];
static bool czReady_ = false;

static void buildCzSign() {
    for (int i = 0; i < kDim; i++){
        float s = 1.0f;
        for (int k = 0; k < VQC_N_CZ; k++) {
            int ma = 1 << (VQC_N_QUBITS - 1 - VQC_CZ[k][0]);
            int mb = 1 << (VQC_N_QUBITS - 1 - VQC_CZ[k][1]);
            if ((i & ma) && (i & mb)) s = -s;
        }
        czSign_[i] = s;
    }
    czReady_ = true;
}

static void applyRot(int q, float phi, float theta, float omega) {
    const float c = cosf(0.5f * theta), s = sinf(0.5f * theta);
    const float ap = 0.5f * (phi + omega), am = 0.5f * (phi - omega);
    const float cap = cosf(ap), sap = sinf(ap), cam = cosf(am), sam = sinf(am);
    const float m00r = cap * c, m00i = -sap * c;
    const float m01r = -cam * s, m01i = -sam * s;
    const float m10r = cam * s, m10i = -sam * s;
    const float m11r = cap * c, m11i = sap * c;
    const int stride = 1 << (VQC_N_QUBITS - 1 - q);
    for (int i = 0; i < kDim; i++) {
        if (i & stride) continue;
        const int j = i | stride;
        const float ar = re_[i], ai = im_[i], br = re_[j], bi = im_[j];
        re_[i] = m00r * ar - m00i * ai + m01r * br - m01i * bi;
        im_[i] = m00r * ai + m00i * ar + m01r * bi + m01i * br;
        re_[j] = m10r * ar - m10i * ai + m11r * br - m11i * bi;
        im_[j] = m10r * ai + m10i * ar + m11r * bi + m11i * br;
    }
}

static void trainableBlock(int k) {
    for (int q = 0; q < VQC_N_QUBITS; q++) {
        applyRot(q, VQC_ROT[k][q][0], VQC_ROT[k][q][1], VQC_ROT[k][q][2]);
    }
    if (VQC_N_CZ > 0) {
        for (int i = 0; i < kDim; i++) {
            re_[i] *= czSign_[i];
            im_[i] *= czSign_[i];
        }
    }
}

void vqc_features_to_angles(const float *features, float *theta) {
    for (int j = 0; j < 3 * VQC_N_QUBITS; j++) {
        if (j >= VQC_N_FEATURES) {
            theta[j] = 0.0f;
            continue;
        }
        float x = features[j];
        if (x < VQC_LO[j]) x = VQC_LO[j];
        if (x > VQC_HI[j]) x = VQC_HI[j];
        theta[j] = VQC_MAX_ANGLE * (x - VQC_LO[j]) / (VQC_HI[j] - VQC_LO[j]);
    }
}

void vqc_expectations(const float *theta, float *z) {
    if (!czReady_) buildCzSign();
    for (int i = 0; i < kDim; i++) {
        re_[i] = 0.0f;
        im_[i] = 0.0f;
    }
    re_[0] = 1.0f;
    trainableBlock(0);
    for (int l = 0; l < VQC_N_LAYERS; l++) {
        for (int q = 0; q < VQC_N_QUBITS; q++) {
            const float a0 = VQC_ENC_W[l][q][0] * theta[3 * q] + VQC_ENC_B[l][q][0];
            const float a1 = VQC_ENC_W[l][q][1] * theta[3 * q + 1] + VQC_ENC_B[l][q][1];
            const float a2 = VQC_ENC_W[l][q][2] * theta[3 * q + 2] + VQC_ENC_B[l][q][2];
            applyRot(q, a0, a1, a2);
        }
        trainableBlock(l + 1);
    }
    for (int o = 0; o < VQC_N_OUT; o++) {
        const int mask = 1 << (VQC_N_QUBITS - 1 - VQC_READOUT[o]);
        float acc = 0.0f;
        for (int i = 0; i < kDim; i++) {
            const float pr = re_[i] * re_[i] + im_[i] * im_[i];
            acc += (i & mask) ? -pr : pr;
        }
        z[o] = acc;
    }
}

float vqc_predict_features(const float *features) {
    float theta[3 * VQC_N_QUBITS];
    float z[VQC_N_OUT];
    vqc_features_to_angles(features, theta);
    vqc_expectations(theta, z);
    float logit = VQC_HEAD_B;
    for (int o = 0; o < VQC_N_OUT; o++) logit += VQC_HEAD_W[o] * z[o];
    return 1.0f / (1.0f + expf(-logit));
}