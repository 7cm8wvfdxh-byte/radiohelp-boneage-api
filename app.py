"""
RadioHelp — Kemik Yaşı API v2.3 (Render Deploy)
ConvNeXt-Small V1, HuggingFace'ten otomatik indirme
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

app = FastAPI(title="RadioHelp Bone Age API", version="2.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cpu"
AGE_MAX = 240.0
MODEL = None
IMG_SIZE = 512

HF_REPO = "TugrulHanSahin/radiohelp-boneage"
HF_FILENAME = "best_convnext_small.pth"
MODEL_PATH = "models/best_convnext_small.pth"

CALIBRATION_TABLE = {
    ('0-4', 'erkek'):   {'mae': 6.25, 'median': 5.56, 'bias': -0.93, 'n': 44},
    ('0-4', 'kız'):     {'mae': 4.30, 'median': 3.69, 'bias': 0.43,  'n': 45},
    ('4-10', 'erkek'):  {'mae': 8.09, 'median': 6.81, 'bias': 3.31,  'n': 278},
    ('4-10', 'kız'):    {'mae': 6.95, 'median': 5.72, 'bias': 2.98,  'n': 417},
    ('10-14', 'erkek'): {'mae': 7.38, 'median': 6.73, 'bias': -5.57, 'n': 570},
    ('10-14', 'kız'):   {'mae': 8.55, 'median': 7.95, 'bias': -6.95, 'n': 325},
    ('14-20', 'erkek'): {'mae': 6.83, 'median': 5.32, 'bias': -3.04, 'n': 152},
    ('14-20', 'kız'):   {'mae': 8.38, 'median': 6.33, 'bias': 0.94,  'n': 61},
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
    MODEL = BoneAgeModel(backbone_name="convnext_small").to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint)
    MODEL.load_state_dict(state, strict=False)
    MODEL.eval()
    print(f"ConvNeXt-Small yüklendi (MAE: {checkpoint.get('best_mae', 'N/A')})")


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


def predict_image(img_np, gender_val):
    gender_tensor = torch.tensor([gender_val]).to(DEVICE)
    predictions = []
    with torch.no_grad():
        for tf in get_tta_transforms():
            img_tensor = tf(image=img_np)['image'].unsqueeze(0).to(DEVICE)
            pred = MODEL(img_tensor, gender_tensor).item() * AGE_MAX
            predictions.append(pred)
    return predictions


def build_response(predictions, gender_key, birth_date=None):
    pred_raw = float(np.mean(predictions))
    pred_std = float(np.std(predictions))

    # Yaş grubu ve kalibrasyon tablosu (ham tahmine göre)
    age_group = get_age_group(pred_raw)
    cal = CALIBRATION_TABLE.get((age_group, gender_key),
                                 {'mae': 7.48, 'median': 6.39, 'bias': -1.95, 'n': 0})

    # Bias düzeltme: model sistematik olarak düşük/yüksek tahmin ediyorsa düzelt
    # Bias -5.57 ise model 5.57 ay düşük tahmin ediyor → +5.57 ekle
    pred_mean = pred_raw - cal['bias']
    
    # Sınırla: 0-228 ay arası (0-19 yaş)
    pred_mean = max(0, min(228, pred_mean))
    
    yil, ay = int(pred_mean // 12), int(pred_mean % 12)

    combined_error = (pred_std * 0.4) + (cal['mae'] * 0.6)
    confidence_95 = round(combined_error * 1.96, 1)

    if combined_error < 5:
        reliability, reliability_label = "high", "Yüksek Güvenilirlik"
    elif combined_error < 8:
        reliability, reliability_label = "medium", "Orta Güvenilirlik"
    else:
        reliability, reliability_label = "low", "Düşük Güvenilirlik"

    gp = get_gp_reference(pred_mean, gender_key)

    result = {
        "success": True,
        "prediction": {
            "bone_age_years": yil,
            "bone_age_months_remainder": ay,
            "bone_age_total_months": round(pred_mean, 1),
            "bone_age_raw_months": round(pred_raw, 1),
            "bone_age_display": f"{yil} yaş {ay} ay",
            "bias_correction": round(cal['bias'], 2),
            "confidence_interval_months": confidence_95,
            "confidence_interval_display": f"±{confidence_95} ay",
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
            "model_type": "ConvNeXt-Small (V1)",
            "version": "2.3",
            "img_size": IMG_SIZE,
            "tta_count": 5,
        },
        "disclaimer": "Bu sonuç yalnızca karar destek amaçlıdır. Kesin tanı için radyolog değerlendirmesi gereklidir."
    }

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
                "chronological_age_years": krono_ay // 12,
                "chronological_age_months_remainder": krono_ay % 12,
                "chronological_age_total_months": krono_ay,
                "chronological_age_display": f"{krono_ay // 12} yaş {krono_ay % 12} ay",
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
        "version": "2.3",
        "model": "ConvNeXt-Small (V1)",
        "mae": "7.48 ay",
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
        return {"success": False, "error": "Model yüklenmedi"}

    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    img_np = np.array(img)

    gender_val = 1.0 if gender.lower() in ['erkek', 'm', 'male', '1'] else 0.0
    gender_key = 'erkek' if gender_val == 1.0 else 'kız'

    predictions = predict_image(img_np, gender_val)
    return build_response(predictions, gender_key, birth_date)


@app.post("/api/bone-age-base64")
async def predict_bone_age_base64(data: dict):
    if MODEL is None:
        return {"success": False, "error": "Model yüklenmedi"}

    import base64
    img_bytes = base64.b64decode(data['image'])
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img)

    gender = data.get('gender', 'erkek')
    gender_val = 1.0 if gender.lower() in ['erkek', 'm', 'male', '1'] else 0.0
    gender_key = 'erkek' if gender_val == 1.0 else 'kız'

    predictions = predict_image(img_np, gender_val)
    return build_response(predictions, gender_key, data.get('birth_date'))
