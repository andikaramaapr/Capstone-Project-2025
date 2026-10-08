# Capstone-Project

# River Water Quality Detection System

An AIoT-based river water quality monitoring system that uses **pH, TDS (Total Dissolved Solids), and turbidity sensors** to collect and classify water quality data in real time.

The system uses an **ESP32** for local data processing and **Decision Tree-based TinyML inference**, allowing water quality classification to be performed directly on the edge device without fully relying on an internet connection. Sensor measurements and classification results are transmitted via **MQTT** to **AWS IoT Core**, stored in **DynamoDB**, and visualized through a web-based dashboard.

## Key Features

- Real-time river water quality monitoring
- pH, TDS, and turbidity sensing
- On-device Decision Tree classification
- TinyML inference on ESP32
- MQTT-based communication
- AWS IoT Core and DynamoDB integration
- Web-based real-time and historical data visualization

## Performance

- **Sensor measurement accuracy:** 80%
- **Classification accuracy:** 65% (13/20 test samples)
- **Decision Tree:** 17 nodes, maximum depth of 5
- **Model size:** approximately 6.2 KB
- **Average inference time:** 14.5 ms
- **Inference platform:** ESP32
