#include <SPI.h>
#include <Arduino_RouterBridge.h>

#include "ads1293.h"
#include "dsp.h"
#include "test_vectors.h"
#include "vqc.h"
#include "vqc_params.h"

const int PIN_CS = 10;
const int PIN_DRDY = 2;
const bool LEAD_OFF_DETECT = true;
const float FS_IN = 102400.0f / (4.0f * 5.0f * 6.0f);
const float FS_OUT = 100.0f;
const float ADC_MAX = 12800000.0f;
const float VREF = 2.4f;
const float INA_GAIN = 3.5f;
const int CHUNK = 25;
const int LOD_EVERY = 32;

ADS1293 ecg(PIN_CS, PIN_DRDY);
Biquad notch50, aa1, aa2;
CubicResampler resampler(FS_IN, FS_OUT);

volatile bool drdyFlag = false;
volatile uint32_t drdyMissed = 0;
float chunk[CHUNK];
int chunkIdx = 0;
uint32_t seq = 0;
bool leadOffInChunk = false;
uint32_t inCount = 0, inPeriod = 0, lastReportMs = 0;
bool sensorOk = false;
volatile int resultCode = 0;
uint32_t lastLedMs = 0;
bool ledOn = false;

void onDrdy() {
  if (drdyFlag) drdyMissed++;
  drdyFlag = true;
}

float codeToMv(uint32_t code) {
  return ((float)code / ADC_MAX - 0.5f) * (2.0f * VREF / INA_GAIN) * 1000.0f;
}

void sendChunk() {
  String s;
  s.reserve(CHUNK * 9 + 24);
  s += seq++;
  s += leadOffInChunk ? ",1" : ",0";
  for (int i = 0; i < CHUNK; i++) {
    s += ",";
    s += String(chunk[i], 4);
  }
  Bridge.notify("ecg_chunk", s);
  chunkIdx = 0;
  leadOffInChunk = false;
}

void processSample() {
  const uint32_t code = ecg.readCh1Raw();
  if (LEAD_OFF_DETECT && (inCount % LOD_EVERY == 0) && ecg.leadOffFlags()) leadOffInChunk = true;
  inCount++;
  inPeriod++;
  float y = biquadRun(notch50, codeToMv(code));
  y = biquadRun(aa2, biquadRun(aa1, y));
  float out[2];
  const int k = resampler.push(y, out);
  for (int i = 0; i < k; i++) {
    chunk[chunkIdx++] = out[i];
    if (chunkIdx == CHUNK) sendChunk();
  }
}

String vqcPredict(String csv) {
  float f[VQC_N_FEATURES];
  int n = 0;
  const char *s = csv.c_str();
  char *end = nullptr;
  while (n < VQC_N_FEATURES) {
    f[n++] = strtof(s, &end);
    if (*end != ',') break;
    s = end + 1;
  }
  if (n != VQC_N_FEATURES) return String("nan,0");
  const uint32_t t0 = micros();
  const float p = vqc_predict_features(f);
  const uint32_t dt = micros() - t0;
  return String(p, 6) + "," + String(dt);
}

void showResult(int code) {
  resultCode = code;
}

void vqcSelfTest() {
  float maxErr = 0.0f;
  uint32_t total = 0;
  for (int i = 0; i < VQC_N_TESTS; i++) {
    const uint32_t t0 = micros();
    const float p = vqc_predict_features(VQC_TEST_X[i]);
    total += micros() - t0;
    const float e = fabsf(p - VQC_TEST_P[i]);
    if (e > maxErr) maxErr = e;
  }
  Monitor.print("VQC self-test: ");
  Monitor.print(VQC_N_TESTS);
  Monitor.print(" vectors, max |dp| = ");
  Monitor.print(maxErr, 8);
  Monitor.print(", mean latency = ");
  Monitor.print(total / (float)VQC_N_TESTS, 1);
  Monitor.println(" us");
}

void updateLed() {
  const uint32_t now = millis();
  uint32_t period;
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
  const uint32_t now = millis();
  if (now - lastReportMs < 10000) return;
  Monitor.print("fs_in=");
  Monitor.print(inPeriod / ((now - lastReportMs) / 1000.0f), 1);
  Monitor.print(" Hz, missed_drdy=");
  Monitor.print(drdyMissed);
  Monitor.print(", seq=");
  Monitor.println(seq);
  inPeriod = 0;
  lastReportMs = now;
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  pinMode(LED_BUILTIN, OUTPUT);
  biquadNotch(notch50, 50.0f, FS_IN, 30.0f);
  biquadLowpass(aa1, 40.0f, FS_IN, 0.5411961f);
  biquadLowpass(aa2, 40.0f, FS_IN, 1.3065630f);
  Bridge.provide("vqc_predict", vqcPredict);
  Bridge.provide("show_result", showResult);
  vqcSelfTest();
  sensorOk = ecg.begin(LEAD_OFF_DETECT);
  Monitor.print("ADS1293 REVID=0x");
  Monitor.print(ecg.revision(), HEX);
  Monitor.println(sensorOk ? " OK" : " NOT FOUND");
  if (sensorOk) attachInterrupt(digitalPinToInterrupt(PIN_DRDY), onDrdy, FALLING);
  else resultCode = 3;
  lastReportMs = millis();
}

void loop() {
  if (sensorOk && drdyFlag) {
    drdyFlag = false;
    processSample();
  }
  updateLed();
  reportStats();
}