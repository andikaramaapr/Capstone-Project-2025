#include <Arduino.h>
#include "SVMClassifier.h"  // pastikan file ini valid, hasil export EloquentML

Eloquent::ML::Port::SVM classifier;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("✅ Model SVM dimuat. Mulai inferensi...");
}

void loop() {
// Jadikan global (heap-safe)
float features[3] = {0, 0, 0};
  
  float ph = 6.5;
  float turbidity = 2.3;
  float solids = 400;

  features[0] = ph;
  features[1] = turbidity;
  features[2] = solids;

  int result = classifier.predict(features);

  if (result == 1) {
    Serial.println("✅ Layak konsumsi");
  } else {
    Serial.println("🚫 Tidak layak konsumsi");
  }

  delay(2000);
}
