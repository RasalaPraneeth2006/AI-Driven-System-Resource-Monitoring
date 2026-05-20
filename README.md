# 🚀 AI-Driven Real-Time System Resource Monitoring and Usage Prediction

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange?style=for-the-badge&logo=pytorch)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-Educational-lightgrey?style=for-the-badge)

</p>

---

# 📖 Project Overview

The **AI-Driven Real-Time System Resource Monitoring and Usage Prediction** project is an intelligent monitoring platform designed to track, analyze, and predict computer system resource utilization in real time.

The system continuously monitors critical performance metrics such as:

- CPU Utilization
- Memory Consumption
- Disk Usage
- Disk I/O Activity
- Battery Health & Status

By integrating **Machine Learning** and **Deep Learning** models, the application not only visualizes current system behavior but also predicts future resource usage patterns and identifies anomalies before performance degradation occurs.

The project combines:
- Real-time system monitoring
- AI-powered predictive analytics
- Interactive visualization dashboards
- Intelligent anomaly detection

to create a proactive and efficient monitoring solution.

---

# ✨ Key Features

✅ Real-time CPU, RAM, and Disk monitoring  
✅ Live battery health tracking  
✅ AI-powered system usage prediction  
✅ Random Forest–based predictive analytics  
✅ LSTM deep learning forecasting model  
✅ Real-time anomaly detection system  
✅ Interactive Streamlit dashboard  
✅ Dynamic Plotly visualizations  
✅ Lightweight and scalable architecture  
✅ Clean dark-themed user interface  

---

# 🧠 AI & Machine Learning Integration

This project leverages both traditional Machine Learning and Deep Learning techniques for intelligent prediction and analysis.

## 🔹 Models Used

| Model | Purpose |
|-------|----------|
| Random Forest Regressor | Predicts future resource utilization |
| LSTM Neural Network | Time-series forecasting of system behavior |

## 🔹 AI Capabilities

- Predictive resource analysis
- Usage pattern recognition
- Performance forecasting
- Intelligent anomaly detection
- Proactive system insights

---

# 🏗️ System Architecture & Data Flow

The project follows a modular monitoring and prediction pipeline:

```text
System Resources
       │
       ▼
Psutil Data Collection
       │
       ▼
Data Preprocessing & Scaling
       │
       ├──────────────► Random Forest Model
       │
       └──────────────► LSTM Prediction Model
                               │
                               ▼
                    Prediction & Anomaly Detection
                               │
                               ▼
                  Streamlit Dashboard Frontend
                               │
                               ▼
                 Interactive Visualization (Plotly)
```

### 🔍 Workflow Summary

1. **Psutil** collects real-time system resource metrics.
2. Data is preprocessed and normalized using scalers.
3. ML and Deep Learning models generate predictions.
4. The anomaly detection module identifies abnormal behavior.
5. Results are visualized on the Streamlit dashboard using Plotly.

---

# 🛠️ Technology Stack

| Category | Technologies |
|----------|--------------|
| **Programming Language** | Python |
| **Frontend Dashboard** | Streamlit |
| **Visualization** | Plotly |
| **Machine Learning** | Scikit-learn |
| **Deep Learning** | PyTorch |
| **Data Processing** | Pandas, NumPy |
| **System Monitoring** | Psutil |
| **Model Serialization** | Pickle (.pkl), PyTorch (.pt) |

---

# 📂 Project Structure

```bash
AI-Driven-System-Resource-Monitoring/
│
├── dashboard.py              # Main Streamlit dashboard application
├── train_model.py           # Model training pipeline
│
├── resource_model.pkl        # Trained Random Forest model
├── scaler.pkl                # Scaler for ML preprocessing
│
├── lstm_model.pt             # Trained LSTM model
├── lstm_scaler.pkl           # LSTM scaler
├── lstm_config.json          # LSTM configuration settings
│
└── README.md                 # Project documentation
```

---

# ⚙️ Installation & Setup

## 🔹 1. Clone the Repository

```bash
git clone https://github.com/RasalaPraneeth2006/AI-Driven-System-Resource-Monitoring.git
```

---

## 🔹 2. Navigate to Project Directory

```bash
cd AI-Driven-System-Resource-Monitoring
```

---

## 🔹 3. Create a Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 🔹 4. Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔹 5. Run the Application

```bash
streamlit run dashboard.py
```

---

# 📊 Monitored System Parameters

The dashboard continuously tracks:

| Parameter | Description |
|-----------|-------------|
| CPU Usage | Processor utilization percentage |
| RAM Usage | Memory consumption analysis |
| Disk Usage | Storage utilization tracking |
| Disk I/O | Read/write activity monitoring |
| Battery Status | Battery percentage & charging status |
| Prediction Metrics | AI-generated future usage estimates |
| Health Indicators | Overall system health analysis |

---

# 📈 Dashboard Functionalities

The dashboard provides:

- 📌 Real-time monitoring cards
- 📌 Live performance graphs
- 📌 AI prediction visualizations
- 📌 Health gauge indicators
- 📌 Anomaly alert notifications
- 📌 Interactive analytical charts
- 📌 Dynamic dark-theme interface

---

# 🎯 Project Objectives

- Develop a real-time monitoring solution for system resources
- Integrate AI-based predictive analytics
- Detect abnormal system behavior proactively
- Improve system performance management
- Build a scalable and interactive dashboard

---

# 🔍 Advantages

✔️ Intelligent automated monitoring  
✔️ AI-enhanced predictive analysis  
✔️ Real-time visualization and analytics  
✔️ Early anomaly detection capability  
✔️ Lightweight and scalable implementation  
✔️ User-friendly dashboard experience  

---

# 🔮 Future Enhancements

- Cloud deployment support
- Multi-device monitoring
- Email & SMS alert integration
- Historical data storage and analytics
- Advanced forecasting models
- Mobile application support
- Database integration

---

# 👨‍💻 Developer

## R. Praneeth

Passionate about:
- Artificial Intelligence
- Machine Learning
- Predictive Analytics
- Real-Time Monitoring Systems
- Data Visualization

---

# 📚 Academic & Research Relevance

This project was developed as part of academic learning and practical implementation in:

- Artificial Intelligence
- Machine Learning
- Predictive Analytics
- Real-Time Data Processing
- Intelligent Monitoring Systems

---

# 📜 License

This project is intended for educational, research, and learning purposes.

---

# ⭐ Conclusion

The **AI-Driven Real-Time System Resource Monitoring and Usage Prediction** system demonstrates how Artificial Intelligence can significantly enhance traditional monitoring solutions through predictive analytics and intelligent decision-making.

By combining:
- Real-time system monitoring
- Machine Learning forecasting
- Deep Learning models
- Interactive visualization

the project delivers a proactive and modern approach to system performance management.
