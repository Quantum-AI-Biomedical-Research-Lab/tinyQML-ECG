#pragma once

float vqc_predict_features(const float *features);
void vqc_expectations(const float *theta, float *z);
void vqc_features_to_angles(const float *features, float *theta);