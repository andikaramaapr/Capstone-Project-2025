#include <EEPROM.h>
#include "GravityTDS.h"
#include "tree_model.h"  // File model hasil ekspor dari Python

// --- PIN CONFIGURATION ---
const int phPin = 26;
const int turbidityPin = 33;
#define TdsSensorPin 14

// --- pH CALIBRATION ---
float voltage4 = 2.362; // Tegangan pada pH 4.00
float voltage7 = 2.057; // Tegangan pada pH 7.00

// --- TDS OBJECT ---
GravityTDS gravityTds;
float temperature = 25.0; // Ubah jika pakai sensor suhu eksternal

// --- ML Normalization Params (Z-score) ---
const float MEAN[] = {7.0808f, 22014.0925f, 3.9668f};
const float STD[]  = {1.4697f, 8767.2324f, 0.7803f};

// Fungsi normalisasi Z-score
float normalize(float value, float mean, float std) {
  return (value - mean) / std;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);

  // --- Init TDS Sensor ---
  gravityTds.setPin(TdsSensorPin);
  gravityTds.setAref(3.3);
  gravityTds.setAdcRange(4095);
  gravityTds.begin();
}

void loop() {
  // --- BACA pH ---
  int phAdc = analogRead(phPin);
  float phVoltage = phAdc * (3.3 / 4095.0);
  float phSlope = (7.00 - 4.00) / (voltage7 - voltage4);
  float phIntercept = 7.00 - (phSlope * voltage7);
  float phValue = (phSlope * phVoltage) + phIntercept;

  // --- BACA TDS ---
  gravityTds.setTemperature(temperature);
  gravityTds.update();
  float tdsValue = gravityTds.getTdsValue();

  // --- BACA TURBIDITY ---
  int turbidityAdc = analogRead(turbidityPin);
  float turbidityValue = map(turbidityAdc, 2800, 0, 5, 1); // Adjust if needed

  // --- NORMALISASI untuk ML ---
  float features[3];
  features[0] = normalize(phValue, MEAN[0], STD[0]);
  features[1] = normalize(tdsValue, MEAN[1], STD[1]);
  features[2] = normalize(turbidityValue, MEAN[2], STD[2]);

  // --- INFERENSI ML ---
  Eloquent::ML::Port::DecisionTree model;
  int prediction = model.predict(features);

  // --- SERIAL OUTPUT ---
  Serial.println("====== Sensor Readings & Prediction ======");
  Serial.print("pH       => ");
  Serial.print(phValue, 2);
  Serial.print(" (V: ");
  Serial.print(phVoltage, 3);
  Serial.println(")");

  Serial.print("TDS      => ");
  Serial.print(tdsValue, 0);
  Serial.println(" ppm");

  Serial.print("Turbidity=> ");
  Serial.print(turbidityAdc);
  Serial.print(" | NTU: ");
  Serial.println(turbidityValue, 2);

  Serial.print("Prediksi => ");
  Serial.println(prediction == 1 ? "Layak ✅" : "Tidak Layak ❌");
  Serial.println("=========================================");
  Serial.println();

  delay(1000); // Delay antar siklus pembacaan + prediksi
}
