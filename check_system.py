import sys
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_gpu():
    print("\n--- 0. Checking GPU / CUDA ---")
    try:
        import torch
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            name = torch.cuda.get_device_name(device_id)
            props = torch.cuda.get_device_properties(device_id)
            vram_gb = props.total_memory / 1e9
            print(f"[OK] CUDA available  ->  {name}  ({vram_gb:.1f} GB VRAM)")
            print(f"     CUDA version: {torch.version.cuda}")
            print(f"     PyTorch version: {torch.__version__}")
        else:
            print("[WARN] CUDA NOT available -- model will run on CPU")
            print(f"       PyTorch version: {torch.__version__}")
            print("       To enable GPU, install CUDA-enabled PyTorch:")
            print("       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        env_device = os.getenv('DEVICE', 'not set')
        print(f"     DEVICE env var: '{env_device}'")
        return torch.cuda.is_available()
    except ImportError:
        print("[FAIL] torch not installed")
        return False

def check_requirements():
    print("\n--- 1. Checking Dependencies ---")
    required = ['flask', 'flask_socketio', 'pymongo', 'redis', 'yfinance', 'torch', 'pandas', 'numpy']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"[OK] {pkg} installed")
        except ImportError:
            print(f"[FAIL] {pkg} MISSING")
            missing.append(pkg)
    return len(missing) == 0

def check_mongodb():
    print("\n--- 2. Checking MongoDB ---")
    try:
        from pymongo import MongoClient
        uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        client = MongoClient(uri, serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        print(f"[OK] MongoDB connected: {uri}")
        return True
    except Exception as e:
        print(f"[FAIL] MongoDB connection failed: {e}")
        print("   (System will run in fallback mode without persistence)")
        return False

def check_yfinance():
    print("\n--- 3. Checking yfinance (Market Data Fallback) ---")
    try:
        import yfinance as yf
        ticker = 'GME'
        print(f"   Fetching {ticker} data...")
        df = yf.download(ticker, period='1d', interval='1m', progress=False)
        if not df.empty:
            print(f"[OK] yfinance working: Fetched {len(df)} rows for {ticker}")
            print(f"   Latest close: ${df['Close'].iloc[-1].item():.2f}")
            return True
        else:
            print(f"[FAIL] yfinance returned empty data for {ticker}")
            return False
    except Exception as e:
        print(f"[FAIL] yfinance failed: {e}")
        return False

def check_model():
    print("\n--- 4. Checking Model File ---")
    model_path = 'models/saved_models/best_model.pth'
    if os.path.exists(model_path):
        print(f"[OK] Model file found: {model_path}")
        try:
            import torch
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                print("[OK] Model checkpoint valid (dictionary format)")
                if 'val_f1' in checkpoint:
                    print(f"   Validation F1: {checkpoint['val_f1']:.4f}")
            else:
                print("[OK] Model checkpoint valid (state_dict format)")
            return True
        except Exception as e:
            print(f"[FAIL] Model load failed: {e}")
            return False
    else:
        print(f"[FAIL] Model file NOT found at {model_path}")
        print("   (Run scripts/04_train_model.py to train)")
        return False

def check_imports():
    print("\n--- 5. Checking Project Imports ---")
    sys.path.insert(0, os.path.abspath('.'))
    try:
        from src.inference.data_fusion_service import DataFusionService
        print("[OK] DataFusionService importable")
    except ImportError as e:
        print(f"[FAIL] DataFusionService import failed: {e}")
        return False
        
    try:
        from src.inference.realtime_predictor import RealtimePredictor
        print("[OK] RealtimePredictor importable")
    except ImportError as e:
        print(f"[FAIL] RealtimePredictor import failed: {e}")
        return False
        
    try:
        from api.app import app, socketio
        print("[OK] Flask app & SocketIO importable")
    except ImportError as e:
        print(f"[FAIL] Flask app import failed: {e}")
        return False
        
    return True

if __name__ == '__main__':
    print(f"System Check started at {datetime.now()}")
    
    steps = [
        check_gpu,
        check_requirements,
        check_mongodb,
        check_yfinance,
        check_model,
        check_imports
    ]
    
    passed = 0
    for step in steps:
        if step():
            passed += 1
            
    print(f"\nSummary: {passed}/{len(steps)} checks passed")
    if passed == len(steps):
        print("[OK] SYSTEM READY TO START")
    else:
        print("[FAIL] SOME CHECKS FAILED - SEE ABOVE")
