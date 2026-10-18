#include <Arduino_RouterBridge.h>

const int PIN_ECG = A0;
const int PIN_LO_P = 10;
const int PIN_LO_N = 11;
const uint32_t FS_RAW = 500;
const uint32_t TS_US = 1000000UL / FS_RAW;
const int DECIM = 5;
const int CHUNK = 25;
const float FC_AA = 40.0f;

struct Biquad {
  float b0, b1, b2, a1, a2, z1, z2;
};

Biquad aa1, aa2;
float chunk[CHUNK];
int chunkIdx = 0;
int decimIdx = 0;
uint32_t seq = 0;
uint32_t nextUs = 0;
bool leadsOffInChunk = false;
bool filterReady = false;
uint32_t overruns = 0;
uint32_t samplesInPeriod = 0;
uint32_t lastReportMs = 0;
volatile int resultCode = 0;
uint32_t lastLedMs = 0;
bool ledOn = false;

void designLowpass(Biquad &f, float fc, float fs, float q) {
  float k = tanf(PI * fc / fs);
  float norm = 1.0f / (1.0f + k / q + k * k);
  f.b0 = k * k * norm;
  f.b1 = 2.0f * f.b0;
  f.b2 = f.b0;
  f.a1 = 2.0f * (k * k - 1.0f) * norm;
  f.a2 = (1.0f - k / q + k * k) * norm;
  f.z1 = f.z2 = 0.0f;
}

void primeBiquad(Biquad &f, float x0) {
  f.z1 = x0 * (1.0f - f.b0);
  f.z2 = x0 * (f.b2 - f.a2);
}

inline float runBiquad(Biquad &f, float x) {
  float y = f.b0 * x + f.z1;
  f.z1 = f.b1 * x - f.a1 * y + f.z2;
  f.z2 = f.b2 * x - f.a2 * y;
  return y;
}

void showResult(int code) {
  resultCode = code;
}

void sendChunk() {
  String s;
  s.reserve(CHUNK * 8 + 24);
  s += seq++;
  s += ",";
  s += leadsOffInChunk ? "1" : "0";
  for (int i = 0; i < CHUNK; i++) {
    s += ",";
    s += String(chunk[i], 1);
  }
  Bridge.notify("ecg_chunk", s);
  chunkIdx = 0;
  leadsOffInChunk = false;
}

void sampleOnce() {
  float raw = (float)analogRead(PIN_ECG);
  if (digitalRead(PIN_LO_P) || digitalRead(PIN_LO_N)) leadsOffInChunk = true;
  if (!filterReady) {
    primeBiquad(aa1, raw);
    primeBiquad(aa2, raw);
    filterReady = true;
  }
  float y = runBiquad(aa2, runBiquad(aa1, raw));
  samplesInPeriod++;
  if (++decimIdx >= DECIM) {
    decimIdx = 0;
    chunk[chunkIdx++] = y;
    if (chunkIdx == CHUNK) sendChunk();
  }
}

void updateLed() {
  uint32_t now = millis();
  uint32_t period = 0;
  switch (resultCode) {
    case 1: period = 500; break;
    case 2: period = 100; break;
    case 3: digitalWrite(LED_BUILTIN, HIGH); return;
    default: digitalWrite(LED_BUILTIN, LOW); return;
  }
  if (now - lastLedMs >= period) {
    lastLedMs = now;
    ledOn = !ledOn;
    digitalWrite(LED_BUILTIN, ledOn ? HIGH : LOW);
  }
}

void reportStats() {
  uint32_t now = millis();
  if (now - lastReportMs >= 10000) {
    Monitor.print("fs_raw=");
    Monitor.print(samplesInPeriod / ((now - lastReportMs) / 1000.0f), 1);
    Monitor.print(" overruns=");
    Monitor.print(overruns);
    Monitor.print(" seq=");
    Monitor.println(seq);
    samplesInPeriod = 0;
    lastReportMs = now;
  }
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  pinMode(PIN_LO_P, INPUT);
  pinMode(PIN_LO_N, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  analogReadResolution(12);
  designLowpass(aa1, FC_AA, FS_RAW, 0.5411961f);
  designLowpass(aa2, FC_AA, FS_RAW, 1.3065630f);
  Bridge.provide("show_result", showResult);
  nextUs = micros();
  lastReportMs = millis();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - nextUs) >= 0) {
    if ((int32_t)(now - nextUs) > (int32_t)(10 * TS_US)) {
      overruns++;
      nextUs = now;
    }
    nextUs += TS_US;
    sampleOnce();
  }
  updateLed();
  reportStats();
}