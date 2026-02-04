# Deployment Guide - BCS Fault Prediction

This guide provides instructions for deploying the BCS Fault Prediction system locally or using Docker.

## 🛠 Local Deployment

### 1. Prerequisites
- Python 3.10+
- Models pre-trained (`bcs_fault_model_*.pkl`)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Backend (FastAPI)
```bash
uvicorn api:app --reload --port 8000
```
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Start Frontend (Streamlit)
In a new terminal:
```bash
streamlit run app_ui.py
```
- Web Interface: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker Deployment (Recommended)

### 1. Build and Run
```bash
docker-compose up --build
```

### 2. Accessing Services
- **Web Interface**: [http://localhost:8501](http://localhost:8501)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Stop Services
```bash
docker-compose down
```

---

## 📡 API Usage Example

### Predict Fault (POST)
**Endpoint**: `/predict`

**Payload**:
```json
{
  "manufacturer": "Empire",
  "amax_cell_temp": 45.5,
  "bmax_cell_temp": 42.1,
  "thermistor1": 40.0,
  "thermistor2": 39.8
}
```

**Response**:
```json
{
  "fault_detected": true,
  "probability": 0.82,
  "confidence_level": "High",
  "manufacturer": "Empire",
  "timestamp": "2024-02-04T08:50:00",
  "features_used": ["AMaxCellTemp", "BMaxCellTemp", "BCSThermistor1", "BCSThermistor2"]
}
```

---

## ☁️ Cloud Deployment
The Docker image is ready for:
- **AWS App Runner** / **AWS ECS**
- **Google Cloud Run**
- **Azure Container Instances**
- **Heroku** (using `heroku.yml`)
