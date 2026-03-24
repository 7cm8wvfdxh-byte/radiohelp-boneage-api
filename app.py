"""
RadioHelp — Kemik Yaşı API (Render Deploy)
FastAPI backend — CBAM model ile inference
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import BoneAgeModelV2
import io
import os
import base64
from datetime import date, datetime

app = FastAPI(title="RadioHelp Bone Age API", version="2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  GLOBAL MODEL (bir kere yükle, bellekte tut)
# ============================================================

DEVICE = "cpu"  # Render free tier CPU only
AGE_MAX = 240.0
MODEL = None
IMG_SIZE = 512

CALIBRATION_TABLE = {
    ('0-4', 'erkek'):   {'mae': 8.39, 'median': 6.81, 'bias': -5.84, 'n': 44},
    ('0-4', 'kız'):     {'mae': 6.54, 'median': 6.03, 'bias': -3.10, 'n': 45},
    ('4-10', 'erkek'):  {'mae': 9.30, 'median': 8.17, 'bias': 1.37,  'n': 278},
    ('4-10', 'kız'):    {'mae': 7.35, 'median': 6.11, 'bias': 3.71,  'n': 417},
    ('10-14', 'erkek'): {'mae': 6.92, 'median': 6.29, 'bias': -4.75, 'n': 570},
    ('10-14', 'kız'):   {'mae': 8.49, 'median': 8.02, 'bias': -6.13, 'n': 325},
    ('14-20', 'erkek'): {'mae': 8.76, 'median': 7.33, 'bias': -4.73, 'n': 152},
    ('14-20', 'kız'):   {'mae': 9.36, 'median': 7.43, 'bias': 3.93,  'n': 61},
}

GP_ATLAS = {
    'erkek': {
        0: "Yenidoğan: Ossifikasyon merkezi görülmez",
        6: "6 ay: Kapitat ve hamat belirgin",
        12: "1 yaş: Distal radius epifizi görülmeye başlar",
        24: "2 yaş: Trikuetrum, lunat belirgin",
        36: "3 yaş: Skafoid, trapezium başlangıcı",
        48: "4 yaş: Tüm karpal kemikler görülür (pisiform hariç)",
        60: "5 yaş: Falangeal epifizler belirginleşir",
        72: "6 yaş: Metakarpal epifizler gelişir",
        84: "7 yaş: Pisiform ossifikasyonu başlar",
        96: "8 yaş: Distal ulna epifizi belirginleşir",
        108: "9 yaş: Epifizler büyümeye devam",
        120: "10 yaş: Karpal kemikler adult şekle yaklaşır",
        132: "11 yaş: Epifiz plakları daralmaya başlar",
        144: "12 yaş: Distal radius epifiz füzyonu başlangıcı",
        156: "13 yaş: Epifiz füzyonları ilerler",
        168: "14 yaş: Çoğu epifiz kapanmış veya kapanmak üzere",
        180: "15 yaş: Distal radius ve ulna füzyonu tamamlanır",
        192: "16 yaş: Tüm epifizler kapanmış",
        204: "17 yaş: Tam skeletal matürite",
    },
    'kız': {
        0: "Yenidoğan: Ossifikasyon merkezi görülmez",
        6: "6 ay: Kapitat ve hamat belirgin, distal radius başlangıcı",
        12: "1 yaş: Lunat, trikuetrum ossifikasyonu",
        24: "2 yaş: Trapezium, trapezoid başlangıcı",
        36: "3 yaş: Tüm karpal kemikler görülür (pisiform hariç)",
        48: "4 yaş: Falangeal epifizler belirginleşir",
        60: "5 yaş: Metakarpal epifizler gelişir",
        72: "6 yaş: Pisiform ossifikasyonu",
        84: "7 yaş: Distal ulna epifizi",
        96: "8 yaş: Karpal kemikler adult şekle yaklaşır",
        108: "9 yaş: Epifiz plakları daralmaya başlar",
        120: "10 yaş: Distal radius epifiz füzyonu başlangıcı",
        132: "11 yaş: Epifiz füzyonları ilerler",
        144: "12 yaş: Çoğu epifiz kapanmış veya kapanmak üzere",
        156: "13 yaş: Distal radius ve ulna füzyonu",
        168: "14 yaş: Tam skeletal matürite",
    }
}


def load_model():
    global MODEL
    model_path = "models/best_convnext_cbam.pth"
    
    if not os.path.exists(model_path):
        # HuggingFace'ten indir (ilk başlatmada)
        print("📥 Model indiriliyor...")
        os.makedirs("models", exist_ok=True)
        # TODO: HuggingFace URL eklenecek
        # urllib.request.urlretrieve(HF_URL, model_path)
    
    MODEL = BoneAgeModelV2().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint)
    MODEL.load_state_dict(state, strict=False)
    MODEL.eval()
    print(f"✅ CBAM model yüklendi (MAE: {checkpoint.get('best_mae', 'N/A')})")


def get_tta_transforms():
    base_norm = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return [
        A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(**base_norm), ToTensorV2()]),
        A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.HorizontalFlip(p=1.0), A.Normalize(**base_norm), ToTensorV2()]),
        A.Compose([A.Resize(int(IMG_SIZE*1.1), int(IMG_SIZE*1.1)), A.CenterCrop(IMG_SIZE, IMG_SIZE), A.Normalize(**base_norm), ToTensorV2()]),
        A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.RandomBrightnessContrast(brightness_limit=(0.1,0.1), contrast_limit=(0.1,0.1), p=1.0), A.Normalize(**base_norm), ToTensorV2()]),
        A.Compose([A.Resize(int(IMG_SIZE*0.95), int(IMG_SIZE*0.95)), A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=0, value=0), A.Normalize(**base_norm), ToTensorV2()]),
    ]


def get_age_group(months):
    if months < 48: return '0-4'
    elif months < 120: return '4-10'
    elif months < 168: return '10-14'
    else: return '14-20'


def get_gp_reference(months, gender):
    atlas = GP_ATLAS.get(gender, GP_ATLAS['erkek'])
    closest = min(atlas.keys(), key=lambda x: abs(x - months))
    return {
        'closest_age_months': closest,
        'closest_age_display': f"{closest // 12} yaş {closest % 12} ay",
        'description': atlas[closest]
    }


# ============================================================
#  STARTUP
# ============================================================

@app.on_event("startup")
async def startup():
    load_model()


# ============================================================
#  ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "RadioHelp Bone Age API",
        "version": "2.3",
        "model": "ConvNeXt-CBAM (V2)",
        "mae": "7.33 ay",
        "status": "ready" if MODEL is not None else "loading"
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/api/bone-age")
async def predict_bone_age(
    image: UploadFile = File(...),
    gender: str = Form(...),
    birth_date: str = Form(None)
):
    """
    Kemik yaşı tahmini.
    
    - image: Sol el AP grafisi (JPEG/PNG)
    - gender: 'erkek' veya 'kız'
    - birth_date: Doğum tarihi YYYY-MM-DD (opsiyonel)
    """
    if MODEL is None:
        return {"success": False, "error": "Model yüklenmedi"}
    
    # Görüntüyü oku
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_np = np.array(img)
    
    gender_val = 1.0 if gender.lower() in ['erkek', 'm', 'male', '1'] else 0.0
    gender_key = 'erkek' if gender_val == 1.0 else 'kız'
    gender_tensor = torch.tensor([gender_val]).to(DEVICE)
    
    # TTA tahmin
    predictions = []
    with torch.no_grad():
        for tf in get_tta_transforms():
            img_tensor = tf(image=img_np)['image'].unsqueeze(0).to(DEVICE)
            pred = MODEL(img_tensor, gender_tensor).item() * AGE_MAX
            predictions.append(pred)
    
    pred_mean = float(np.mean(predictions))
    pred_std = float(np.std(predictions))
    yil, ay = int(pred_mean // 12), int(pred_mean % 12)
    
    # Kalibre güven aralığı
    age_group = get_age_group(pred_mean)
    cal = CALIBRATION_TABLE.get((age_group, gender_key),
                                 {'mae': 7.33, 'median': 6.21, 'bias': -1.03, 'n': 0})
    
    combined_error = (pred_std * 0.4) + (cal['mae'] * 0.6)
    confidence_95 = round(combined_error * 1.96, 1)
    
    if combined_error < 5:
        reliability = "high"
        reliability_label = "Yüksek Güvenilirlik"
    elif combined_error < 8:
        reliability = "medium"
        reliability_label = "Orta Güvenilirlik"
    else:
        reliability = "low"
        reliability_label = "Düşük Güvenilirlik"
    
    # GP Atlas
    gp = get_gp_reference(pred_mean, gender_key)
    
    # Response
    result = {
        "success": True,
        "prediction": {
            "bone_age_months": round(pred_mean, 1),
            "bone_age_display": f"{yil} yıl {ay} ay",
            "confidence_interval": f"±{confidence_95} ay",
            "tta_std": round(pred_std, 2),
            "reliability": reliability,
            "reliability_label": reliability_label,
            "combined_error": round(combined_error, 2),
        },
        "calibration": {
            "age_group": f"{age_group} Yaş",
            "group_mae": cal['mae'],
            "group_median": cal['median'],
            "group_bias": cal['bias'],
            "group_n": cal['n'],
        },
        "atlas_comparison": gp,
        "model_info": {
            "model_type": "ConvNeXt-CBAM (V2)",
            "version": "2.3",
            "img_size": IMG_SIZE,
            "tta_count": 5,
        },
        "disclaimer": "Bu sonuç yalnızca karar destek amaçlıdır. Kesin tanı için radyolog değerlendirmesi gereklidir."
    }
    
    # Kronolojik yaş (varsa)
    if birth_date:
        try:
            birth = datetime.strptime(birth_date, '%Y-%m-%d').date()
            today = date.today()
            krono_ay = (today.year - birth.year) * 12 + (today.month - birth.month)
            fark = pred_mean - krono_ay
            
            if abs(fark) <= 12:
                fark_yorum = "Normal sınırlar içinde (±1 yıl)"
            elif abs(fark) <= 24:
                fark_yorum = "Sınırda — klinik değerlendirme önerilir"
            else:
                fark_yorum = "Anormal — endokrinoloji konsültasyonu önerilir"
            
            result["clinical_context"] = {
                "chronological_age_months": krono_ay,
                "chronological_age_display": f"{krono_ay // 12} yıl {krono_ay % 12} ay",
                "difference_months": round(fark, 1),
                "difference_direction": "ileri" if fark > 0 else "geri",
                "assessment": fark_yorum,
                "within_normal": abs(fark) <= 24,
            }
        except ValueError:
            pass
    
    return result


@app.post("/api/bone-age-base64")
async def predict_bone_age_base64(data: dict):
    """
    Base64 encoded görüntü ile tahmin.
    Body: {"image": "base64...", "gender": "erkek", "birth_date": "2015-10-15"}
    """
    if MODEL is None:
        return {"success": False, "error": "Model yüklenmedi"}
    
    import base64
    img_bytes = base64.b64decode(data['image'])
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img)
    
    gender = data.get('gender', 'erkek')
    gender_val = 1.0 if gender.lower() in ['erkek', 'm', 'male', '1'] else 0.0
    gender_key = 'erkek' if gender_val == 1.0 else 'kız'
    gender_tensor = torch.tensor([gender_val]).to(DEVICE)
    
    predictions = []
    with torch.no_grad():
        for tf in get_tta_transforms():
            img_tensor = tf(image=img_np)['image'].unsqueeze(0).to(DEVICE)
            pred = MODEL(img_tensor, gender_tensor).item() * AGE_MAX
            predictions.append(pred)
    
    pred_mean = float(np.mean(predictions))
    pred_std = float(np.std(predictions))
    yil, ay = int(pred_mean // 12), int(pred_mean % 12)
    
    age_group = get_age_group(pred_mean)
    cal = CALIBRATION_TABLE.get((age_group, gender_key),
                                 {'mae': 7.33, 'median': 6.21, 'bias': -1.03, 'n': 0})
    combined_error = (pred_std * 0.4) + (cal['mae'] * 0.6)
    confidence_95 = round(combined_error * 1.96, 1)
    
    if combined_error < 5:
        reliability, label = "high", "Yüksek Güvenilirlik"
    elif combined_error < 8:
        reliability, label = "medium", "Orta Güvenilirlik"
    else:
        reliability, label = "low", "Düşük Güvenilirlik"
    
    gp = get_gp_reference(pred_mean, gender_key)
    
    return {
        "success": True,
        "prediction": {
            "bone_age_months": round(pred_mean, 1),
            "bone_age_display": f"{yil} yıl {ay} ay",
            "confidence_interval": f"±{confidence_95} ay",
            "tta_std": round(pred_std, 2),
            "reliability": reliability,
            "reliability_label": label,
        },
        "calibration": {
            "age_group": f"{age_group} Yaş",
            "group_mae": cal['mae'],
            "group_median": cal['median'],
        },
        "atlas_comparison": gp,
        "disclaimer": "Bu sonuç yalnızca karar destek amaçlıdır."
    }
