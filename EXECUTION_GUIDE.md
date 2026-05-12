# Stock Manipulation Detection System - Complete Execution Guide

## 📋 Prerequisites

✅ Python 3.12 installed
✅ MongoDB installed and running
✅ All dependencies installed (`pip install -r requirements.txt`)

---

## 🚀 Complete Execution Steps

### **Option 1: Quick Start (Demo Mode - No Training Required)**

This runs the system with rule-based analysis (no ML model needed):

```powershell
# Step 1: Start MongoDB (in a new terminal)
mongod

# Step 2: Start Flask API (in another terminal)
cd C:\Users\Kamal\Desktop\Project\stock-manipulation-detector
python api/app.py

# Step 3: Open Dashboard
# Open frontend/index.html in your browser
```

**What you get:**
- ✅ Working API with all endpoints
- ✅ Interactive dashboard
- ✅ Rule-based manipulation detection
- ✅ Real-time predictions
- ❌ No ML model (uses heuristics instead)

---

### **Option 2: Full System with ML Model**

This trains the ML model for better predictions:

#### **Step 1: Collect Historical Data**

```powershell
# Collect 3 months of market data for monitored stocks
python scripts/02_collect_historical_data.py --months 3
```

**Expected output:**
```
Processing GME...
  Collected 395 records
Processing AMC...
  Collected 395 records
...
```

#### **Step 2: Preprocess Data**

```powershell
# Generate manipulation labels
python scripts/03_preprocess_data.py
```

**Expected output:**
```
Total samples: 2374
Manipulation events: 261
Normal events: 2113
Manipulation rate: 11.0%  ✓ (Should be 5-15%)
```

#### **Step 3: Train Model**

```powershell
# Train for 10 epochs (takes 5-10 minutes)
python scripts/04_train_model.py --epochs 10
```

**Expected output:**
```
Epoch 1/10
  Train Loss: 0.4523
  Val Loss: 0.4201
  Val Accuracy: 89.23%
...
✓ Saved best model
```

#### **Step 4: Start Services**

```powershell
# Terminal 1: Start MongoDB
mongod

# Terminal 2: Start Flask API
cd C:\Users\Kamal\Desktop\Project\stock-manipulation-detector
python api/app.py

# Terminal 3 (Optional): Start background services
python scripts/main_service.py
```

#### **Step 5: Open Dashboard**

Open `frontend/index.html` in your browser

---

## 🧪 Testing the System

### **Test API Endpoints**

```powershell
# Health check
curl http://localhost:5000/health

# Get monitored stocks
curl http://localhost:5000/api/stocks

# Get stock details
curl http://localhost:5000/api/stocks/GME

# Make prediction
curl http://localhost:5000/api/stocks/GME/prediction

# Get alerts
curl http://localhost:5000/api/alerts

# Get analysis
curl http://localhost:5000/api/analysis/GME/overview
```

### **Expected Responses**

**Health Check:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-17T18:00:00"
}
```

**Stock Prediction:**
```json
{
  "ticker": "GME",
  "risk_level": "medium",
  "risk_score": 45.2,
  "manipulation_probability": 0.452,
  "model_version": "v1.0"
}
```

---

## 📊 Dashboard Features

Once you open `frontend/index.html`:

1. **Stock Cards** - Real-time risk indicators for each monitored stock
2. **Color-Coded Badges** - Low (green), Medium (yellow), High (orange), Critical (red)
3. **Alert Feed** - Recent manipulation alerts
4. **Live Updates** - WebSocket connection for real-time data
5. **Auto-Refresh** - Updates every 30 seconds

---

## ⚙️ Configuration

Edit `.env` to customize:

```env
# Monitored stocks
MONITORED_TICKERS=GME,AMC,TSLA,AAPL,MSFT,NVDA

# Model settings
MODEL_PATH=models/saved_models/best_model.pth
DEVICE=cpu

# Database
MONGO_URI=mongodb://localhost:27017/
REDIS_URL=redis://localhost:6379  # Optional
```

---

## 🔧 Troubleshooting

### **Issue: MongoDB connection error**
```
Solution: Start MongoDB with 'mongod' command
```

### **Issue: Model not found**
```
Solution: Either:
1. Train the model: python scripts/04_train_model.py
2. Use demo mode (system works without model)
```

### **Issue: No data for preprocessing**
```
Solution: Collect data first:
python scripts/02_collect_historical_data.py --months 3
```

### **Issue: 99% manipulation rate**
```
Solution: Already fixed! Should now show 5-15%
If still high, check labeler thresholds in:
src/preprocessing/labeler.py
```

### **Issue: API errors for social media**
```
Solution: This is normal without API credentials
System works in demo mode without live data
See DEMO_MODE.md for details
```

---

## 📁 Project Structure

```
stock-manipulation-detector/
├── api/                    # Flask API
│   ├── app.py             # Main application
│   └── routes/            # API endpoints
├── src/
│   ├── data_collection/   # Data collectors
│   ├── preprocessing/     # Data preprocessing
│   ├── models/            # ML models
│   └── inference/         # Predictions & alerts
├── scripts/               # Execution scripts
│   ├── 02_collect_historical_data.py
│   ├── 03_preprocess_data.py
│   └── 04_train_model.py
├── frontend/              # Dashboard
│   └── index.html
├── data/                  # Data storage
│   ├── raw/
│   ├── processed/
│   └── labeled/
└── models/                # Saved models
    └── saved_models/
```

---

## 🎯 Quick Reference

### **Minimal Setup (No Training)**
```powershell
mongod                    # Terminal 1
python api/app.py         # Terminal 2
# Open frontend/index.html
```

### **Full Setup (With Training)**
```powershell
python scripts/02_collect_historical_data.py --months 3
python scripts/03_preprocess_data.py
python scripts/04_train_model.py --epochs 10
mongod                    # Terminal 1
python api/app.py         # Terminal 2
# Open frontend/index.html
```

### **Stop Services**
```powershell
Ctrl+C  # In each terminal
```

---

## ✅ Success Indicators

**Preprocessing:**
- ✅ Manipulation rate: 5-15%
- ❌ Manipulation rate: >90% (too high)

**Training:**
- ✅ Validation accuracy: >85%
- ✅ Model saved successfully

**API:**
- ✅ Health endpoint returns 200
- ✅ Predictions return risk scores

**Dashboard:**
- ✅ Stock cards display
- ✅ WebSocket connected
- ✅ Real-time updates working

---

## 📚 Additional Documentation

- `QUICK_START.md` - Quick start guide
- `DEMO_MODE.md` - Running without API credentials
- `COMPLETION_GUIDE.md` - Detailed setup guide
- `walkthrough.md` - Complete system documentation

---

## 🎉 You're Ready!

The system is now fully operational. Choose your execution path:
- **Quick Demo**: Option 1 (5 minutes)
- **Full System**: Option 2 (30-45 minutes including training)

Both options provide a fully functional manipulation detection system!
