#pragma once
#include <math.h>
#include <stdint.h>

static const float kPi = 3.14159265358979f;

struct Biquad {
    float b0, b1, b2, a1, a2, z1, z2;
    bool primed;
};

inline void biquadLowpass(Biquad &f, float fc, float fs, float q) {
    const float k = tanf(kPi * fc / fs);
    const float norm = 1.0f / (1.0f + k / q + k * k);
    f.b0 = k * k * norm;
    f.b1 = 2.0f * f.b0;
    f.b2 = f.b0;
    f.a1 = 2.0f * (k * k - 1.0f) * norm;
    f.a2 = (1.0f - k / q + k * k) * norm;
    f.z1 = f.z2 = 0.0f;
    f.primed = false;
}

inline void biquadNotch(Biquad &f, float f0, float fs, float q) {
    const float w0 = 2.0f * kPi * f0 / fs;
    const float alpha = sinf(w0) / (2.0f * q);
    const float cw = cosf(w0);
    const float a0 = 1.0f + alpha;
    f.b0 = 1.0f / a0;
    f.b1 = -2.0f * cw / a0;
    f.b2 = 1.0f / a0;
    f.a1 = -2.0f * cw / a0;
    f.a2 = (1.0f - alpha) / a0;
    f.z1 = f.z2 = 0.0f;
    f.primed = false;
}

inline float biquadRun(Biquad &f, float x) {
    if (!f.primed) {
        f.z1 = x * (1.0f - f.b0);
        f.z2 = x * (f.b2 - f.a2);
        f.primed = true;
    }
    const float y = f.b0 * x + f.z1;
    f.z1 = f.b1 * x - f.a1 * y + f.z2;
    f.z2 = f.b2 * x - f.a2 * y;
    return y;
}

class CubicResampler {
    public:
        CubicResampler(float fsIn, float fsOut): step_(fsIn / fsOut) {}
        int push(float x, float *out) {
            s_[0] = s_[1];
            s_[1] = s_[2];
            s_[2] = s_[3];
            s_[3] = x;
            if (filled_ < 4) {
                if (++filled_ < 4) return 0;
                phase_ = 0.0f;
            } else {
                phase_ -= 1.0f;
            }
            int k = 0;
            while (phase_ < 1.0f) {
                out[k++] = catmullRom(phase_ < 0.0f ? 0.0f : phase_);
                phase_ += step_;
            }
            return k;
        }

    private:
        float catmullRom(float mu) const {
            const float p0 = s_[0], p1 = s_[1], p2 = s_[2], p3 = s_[3];
            const float a0 = -0.5f * p0 + 1.5f * p1 - 1.5f * p2 + 0.5f * p3;
            const float a1 = p0 - 2.5f * p1 + 2.0f * p2 - 0.5f * p3;
            const float a2 = -0.5f * p0 + 0.5f * p2;
            return ((a0 * mu + a1) * mu + a2) * mu + p1;
        }
        float s_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float step_;
        float phase_ = 0.0f;
        uint8_t filled_ = 0;
};