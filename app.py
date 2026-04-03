"""
RadioHelp — Kemik Yasi API v3.0 (Render Deploy)
ConvNeXt V1 Base, MAE 6.76 ay, HuggingFace'ten otomatik indirme
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import BoneAgeModel
import io
import os
from datetime import date, datetime
from huggingface_hub import hf_hub_download

app = FastAPI(title="RadioHelp Bone Age API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"
MODEL = None
IMG_SIZE = 512

# z-score denormalizasyon parametreleri
AGE_MEAN = 127.3
AGE_STD = 41.2

HF_REPO = "TugrulHanSahin/radiohelp-boneage"
HF_FILENAME = "masked_v1base_fold0_mae6.76.pth"
MODEL_PATH = "models/masked_v1base_fold0_mae6.76.pth"

# Kalibrasyon tablosu (2515 validation orneginden hesaplandi, bias correction YOK)
CALIBRATION_TABLE = {
    ('0-4', 'erkek'):   {'mae': 5.51, 'median': 3.43, 'n': 54},
    ('0-4', 'kiz'):     {'mae': 6.06, 'median': 5.42, 'n': 46},
    ('4-10', 'erkek'):  {'mae': 8.35, 'median': 6.31, 'n': 331},
    ('4-10', 'kiz'):    {'mae': 6.90, 'median': 5.59, 'n': 470},
    ('10-14', 'erkek'): {'mae': 6.19, 'median': 5.12, 'n': 679},
    ('10-14', 'kiz'):   {'mae': 7.07, 'median': 6.00, 'n': 537},
    ('14-20', 'erkek'): {'mae': 5.25, 'median': 4.12, 'n': 277},
    ('14-20', 'kiz'):   {'mae': 7.90, 'median': 6.62, 'n': 121},
}

GP_ATLAS = {
    'erkek': {
        0: "Yenidogan: Ossifikasyon merkezi gorulmez",
        6: "6 ay: Kapitat ve hamat belirgin",
        12: "1 yas: Distal radius epifizi gorulmeye baslar",
        24: "2 yas: Trikuetrum, lunat belirgin",
        36: "3 yas: Skafoid, trapezium baslangici",
        48: "4 yas: Tum karpal kemikler gorulur (pisiform haric)",
        60: "5 yas: Falangeal epifizler belirginlesir",
        72: "6 yas: Metakarpal epifizler gelisir",
        84: "7 yas: Pisiform ossifikasyonu baslar",
        96: "8 yas: Distal ulna epifizi belirginlesir",
        108: "9 yas: Epifizler buyumeye devam",
        120: "10 yas: Karpal kemikler adult sekle yaklasir",
        132: "11 yas: Epifiz plaklari daralmaya baslar",
        144: "12 yas: Distal radius epifiz fuzyonu baslangici",
        156: "13 yas: Epifiz fuzyonlari ilerler",
        168: "14 yas: Cogu epifiz kapanmis veya kapanmak uzere",
        180: "15 yas: Distal radius ve ulna fuzyonu tamamlanir",
        192: "16 yas: Tum epifizler kapanmis",
        204: "17 yas: Tam skeletal maturite",
    },
    'kiz': {
        0: "Yenidogan: Ossifikasyon merkezi gorulmez",
        6: "6 ay: Kapitat ve hamat belirgin, distal radius baslangici",
        12: "1 yas: Lunat, trikuetrum ossifikasyonu",
        24: "2 yas: Trapezium, trapezoid baslangici",
        36: "3 yas: Tum karpal kemikler gorulur (pisiform haric)",
        48: "4 yas: Falangeal epifizler belirginlesir",
        60: "5 yas: Metakarpal epifizler gelisir",
        72: "6 yas: Pisiform ossifikasyonu",
        84: "7 yas: Distal ulna epifizi",
        96: "8 yas: Karpal kemikler adult sekle yaklasir",
        108: "9 yas: Epifiz plaklari daralmaya baslar",
        120: "10 yas: Distal radius epifiz fuzyonu baslangici",
        132: "11 yas: Epifiz fuzyonlari ilerler",
        144: "12 yas: Cogu epifiz kapanmis veya kapanmak uzere",
        156: "13 yas: Distal radius ve ulna fuzyonu",
        168: "14 yas: Tam skeletal maturite",
    }
}


def download_model():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Model indiriliyor: {HF_REPO}/{HF_FILENAME}")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILENAME,
            local_dir="models",
            token=os.environ.get("HF_TOKEN", None)
        )
        print(f"Model indirildi")
    else:
        print(f"Model mevcut: {MODEL_PATH}")


def load_model():
    global MODEL
    download_model()
    MODEL = BoneAgeModel().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    MODEL.load_state_dict(checkpoint['model_state_dict'])
    MODEL.eval()
    print(f"ConvNeXt V1 Base yuklendi (MAE: {checkpoint.get('best_mae', 'N/A')})")


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
        'closest_age_display': f"{closest // 12} yas {closest % 12} ay",
        'description': atlas[closest]
    }


def predict_image(img_np, gender_val):
    gender_tensor = torch.tensor([gender_val]).to(DEVICE)
    predictions = []
    with torch.no_grad():
        for tf in get_tta_transforms():
            img_tensor = tf(image=img_np)['image'].unsqueeze(0).to(DEVICE)
            raw = MODEL(img_tensor, gender_tensor).item()
            # z-score denormalizasyon
            pred_months = raw * AGE_STD + AGE_MEAN
            predictions.append(pred_months)
    return predictions


def build_response(predictions, gender_key, birth_date=None):
    pred_mean = float(np.mean(predictions))
    pred_std = float(np.std(predictions))

    # Clip: 0-228 ay arasi
    pred_mean = max(0, min(228, pred_mean))

    age_group = get_age_group(pred_mean)
    cal = CALIBRATION_TABLE.get((age_group, gender_key),
                                 {'mae': 6.76, 'median': 5.50, 'n': 0})

    yil, ay = int(pred_mean // 12), int(pred_mean % 12)

    combined_error = (pred_std * 0.4) + (cal['mae'] * 0.6)
    confidence_95 = round(combined_error * 1.96, 1)

    if combined_error < 5:
        reliability, reliability_label = "high", "Yuksek Guvenilirlik"
    elif combined_error < 8:
        reliability, reliability_label = "medium", "Orta Guvenilirlik"
    else:
        reliability, reliability_label = "low", "Dusuk Guvenilirlik"

    gp = get_gp_reference(pred_mean, gender_key)

    result = {
        "success": True,
        "prediction": {
            "bone_age_years": yil,
            "bone_age_months_remainder": ay,
            "bone_age_total_months": round(pred_mean, 1),
            "bone_age_display": f"{yil} yas {ay} ay",
            "confidence_interval_months": confidence_95,
            "confidence_interval_display": f"\u00b1{confidence_95} ay",
            "tta_std": round(pred_std, 2),
            "reliability": reliability,
            "reliability_label": reliability_label,
            "combined_error": round(combined_error, 2),
        },
        "calibration": {
            "age_group": f"{age_group} Yas",
            "group_mae": cal['mae'],
            "group_median": cal['median'],
            "group_n": cal['n'],
        },
        "atlas_comparison": gp,
        "model_info": {
            "model_type": "ConvNeXt V1 Base (Masked Pipeline)",
            "version": "3.0",
            "mae": "6.76 ay",
            "img_size": IMG_SIZE,
            "tta_count": 5,
            "normalization": "z-score (mean=127.3, std=41.2)",
        },
        "disclaimer": "Bu sonuc yalnizca karar destek amaclidir. Kesin tani icin radyolog degerlendirmesi gereklidir."
    }

    if birth_date:
        try:
            birth = datetime.strptime(birth_date, '%Y-%m-%d').date()
            today = date.today()
            krono_ay = (today.year - birth.year) * 12 + (today.month - birth.month)
            fark = pred_mean - krono_ay

            if abs(fark) <= 12:
                fark_yorum = "Normal sinirlar icinde (\u00b11 yil)"
            elif abs(fark) <= 24:
                fark_yorum = "Sinirda — klinik degerlendirme onerilir"
            else:
                fark_yorum = "Anormal — endokrinoloji konsultasyonu onerilir"

            result["clinical_context"] = {
                "chronological_age_years": krono_ay // 12,
                "chronological_age_months_remainder": krono_ay % 12,
                "chronological_age_total_months": krono_ay,
                "chronological_age_display": f"{krono_ay // 12} yas {krono_ay % 12} ay",
                "difference_months": round(fark, 1),
                "difference_display": f"{abs(round(fark, 0)):.0f} ay {'ileri' if fark > 0 else 'geri'}",
                "difference_direction": "ileri" if fark > 0 else "geri",
                "assessment": fark_yorum,
                "within_normal": abs(fark) <= 24,
            }
        except ValueError:
            pass

    return result


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/")
async def root():
    return {
        "service": "RadioHelp Bone Age API",
        "version": "3.0",
        "model": "ConvNeXt V1 Base (Masked Pipeline)",
        "mae": "6.76 ay",
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
    if MODEL is None:
        return {"success": False, "error": "Model yuklenmedi"}

    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_np = np.array(img)

    gender_val = 1.0 if gender.lower() in ['erkek', 'm', 'male', '1'] else 0.0
    gender_key = 'erkek' if gender_val == 1.0 else 'kiz'

    predictions = predict_image(img_np, gender_val)
    return build_response(predictions, gender_key, birth_date)


@app.post("/api/bone-age-base64")
async def predict_bone_age_base64(data: dict):
    if MODEL is None:
        return {"success": False, "error": "Model yuklenmedi"}

    import base64
    img_bytes = base64.b64decode(data['image'])
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img)

    gender = data.get('gender', 'erkek')
    gender_val = 1.0 if gender.lower() in ['erkek', 'm', 'male', '1'] else 0.0
    gender_key = 'erkek' if gender_val == 1.0 else 'kiz'

    predictions = predict_image(img_np, gender_val)
    return build_response(predictions, gender_key, data.get('birth_date'))
