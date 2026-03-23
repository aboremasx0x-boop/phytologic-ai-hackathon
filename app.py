from fastapi import FastAPI, UploadFile, File, Form, Header
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import os
import io
import json
import uuid
import hashlib
import secrets
from datetime import datetime
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import os
import requests

MODELS = {
    "plant_disease_model_v3.pth": "https://github.com/aboremasx0x-boop/phytologic-ai-hackathon/releases/download/v1/plant_disease_model_v3.pth",
    "plant_model.pth": "https://github.com/aboremasx0x-boop/phytologic-ai-hackathon/releases/download/v1/plant_model.pth",
    "tomato_disease_model.pth": "https://github.com/aboremasx0x-boop/phytologic-ai-hackathon/releases/download/v1/tomato_disease_model.pth"
}

def download_model(filename, url):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")

        # تحميل على شكل chunks (مهم للملفات الكبيرة)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        print(f"{filename} downloaded successfully")
    else:
        print(f"{filename} already exists")

# تحميل كل الموديلات
for name, link in MODELS.items():
    download_model(name, link)
# =========================================================
# إعداد التطبيق
# =========================================================
app = FastAPI(title="Phytologic AI - Final Global System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# المسارات
# =========================================================
PLANT_MODEL_PATH = "plant_classifier.pth" if os.path.exists("plant_classifier.pth") else "plant_classifier_v1.pth"
PLANT_CLASSES_PATH = "plant_classes.json"

GENERAL_MODEL_PATH = "plant_disease_model.pth" if os.path.exists("plant_disease_model.pth") else "plant_model.pth"
GENERAL_CLASSES_PATH = "disease_classes.json" if os.path.exists("disease_classes.json") else "classes.json"

TOMATO_MODEL_PATH = "tomato_disease_model.pth"
TOMATO_CLASSES_PATH = "tomato_classes.json"

LOG_FILE = "logs.json"
USERS_FILE = "users.json"
SESSIONS_FILE = "sessions.json"

FRONTEND_DIR = "frontend"
STATIC_DIR = "static"
GRADCAM_DIR = os.path.join(STATIC_DIR, "gradcam")

INDEX_PATH = os.path.join(FRONTEND_DIR, "index.html")
DASHBOARD_PATH = os.path.join(FRONTEND_DIR, "dashboard.html")
NATIONAL_CENTER_PATH = os.path.join(FRONTEND_DIR, "national_center.html")
ALERTS_PATH = os.path.join(FRONTEND_DIR, "alerts.html")
FORECAST_PATH = os.path.join(FRONTEND_DIR, "forecast.html")
FORECAST_AI_PATH = os.path.join(FRONTEND_DIR, "forecast_ai.html")
ADMIN_PATH = os.path.join(FRONTEND_DIR, "admin.html")
OPS_PATH = os.path.join(FRONTEND_DIR, "ops.html")
MAP_PATH = os.path.join(FRONTEND_DIR, "map.html")
REPORTS_CENTER_PATH = os.path.join(FRONTEND_DIR, "reports_center.html")
LAYOUT_SHELL_PATH = os.path.join(FRONTEND_DIR, "layout_shell.html")
LOGIN_PATH = os.path.join(FRONTEND_DIR, "login.html")
REGISTER_PATH = os.path.join(FRONTEND_DIR, "register.html")
FARMERS_PATH = os.path.join(FRONTEND_DIR, "farmers.html")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =========================================================
# التحقق من الملفات
# =========================================================
required_files = [
    PLANT_MODEL_PATH,
    GENERAL_MODEL_PATH,
    TOMATO_MODEL_PATH,
]

for fp in required_files:
    if not os.path.exists(fp):
        print(f"Model not found yet: {fp} (will download automatically)")

# =========================================================
# ثوابت وقواميس
# =========================================================
DEFAULT_PLANT_CLASSES = [
    "Apple",
    "Blueberry",
    "Cherry",
    "Corn",
    "Grape",
    "Orange",
    "Peach",
    "Pepper",
    "Potato",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
    "Tomato",
]

ARABIC_PLANT_NAMES = {
    "Apple": "تفاح",
    "Blueberry": "توت أزرق",
    "Cherry": "كرز",
    "Corn": "ذرة",
    "Grape": "عنب",
    "Orange": "برتقال",
    "Peach": "خوخ",
    "Pepper": "فلفل",
    "Potato": "بطاطس",
    "Raspberry": "توت العليق",
    "Soybean": "فول الصويا",
    "Squash": "قرع",
    "Strawberry": "فراولة",
    "Tomato": "طماطم",
    "unknown": "غير معروف",
}

ARABIC_DISEASE_NAMES = {
    "healthy": "سليم",
    "Early_blight": "اللفحة المبكرة",
    "Late_blight": "اللفحة المتأخرة",
    "Leaf_Mold": "عفن الأوراق",
    "Septoria_leaf_spot": "تبقع السبتوريا",
    "Target_Spot": "تبقع الهدف",
    "Tomato_mosaic_virus": "فيروس موزاييك الطماطم",
    "Tomato_Yellow_Leaf_Curl_Virus": "فيروس تجعد واصفرار أوراق الطماطم",
    "Bacterial_spot": "تبقع بكتيري",
    "Spider_mites_Two_spotted_spider_mite": "العنكبوت الأحمر",
    "Common_rust": "الصدأ الشائع",
    "Northern_Leaf_Blight": "لفحة الأوراق الشمالية",
    "Cercospora_leaf_spot_Gray_leaf_spot": "تبقع سركسبورا / التبقع الرمادي",
    "Leaf_blight_Isariopsis_Leaf_Spot": "لفحة الأوراق",
    "Black_rot": "العفن الأسود",
    "Esca_Black_Measles": "إسكا / الحصبة السوداء",
    "Apple_scab": "جرب التفاح",
    "Powdery_mildew": "البياض الدقيقي",
    "Leaf_scorch": "احتراق الأوراق",
    "unknown": "غير معروف",
}

CROP_DISEASE_MAP = {
    "Tomato": [
        "Early_blight",
        "Late_blight",
        "Leaf_Mold",
        "Septoria_leaf_spot",
        "Target_Spot",
        "Tomato_mosaic_virus",
        "Tomato_Yellow_Leaf_Curl_Virus",
        "Bacterial_spot",
        "Spider_mites_Two_spotted_spider_mite",
        "healthy",
    ],
    "Potato": ["Early_blight", "Late_blight", "healthy"],
    "Pepper": ["Bacterial_spot", "healthy"],
    "Corn": ["Common_rust", "Northern_Leaf_Blight", "Cercospora_leaf_spot_Gray_leaf_spot", "healthy"],
    "Grape": ["Black_rot", "Esca_Black_Measles", "Leaf_blight_Isariopsis_Leaf_Spot", "healthy"],
    "Apple": ["Apple_scab", "Black_rot", "Powdery_mildew", "healthy"],
    "Strawberry": ["Leaf_scorch", "healthy"],
}

BIO_PROGRAMS = {
    "Early_blight": {
        "title": "برنامج حيوي داعم للّفحة المبكرة",
        "steps": [
            "إزالة الأوراق الشديدة الإصابة والتخلص منها خارج الحقل.",
            "تحسين التهوية وخفض البلل الورقي.",
            "استخدام منتجات حيوية تحتوي على Bacillus subtilis أو Trichoderma spp.",
            "إعادة التقييم بعد 5–7 أيام.",
        ],
        "materials": [
            {"type": "Biological", "name": "Bacillus subtilis-based product", "dose_note": "اتبع الملصق المسجل."},
            {"type": "Biological", "name": "Trichoderma spp.-based product", "dose_note": "اتبع الملصق المسجل."},
        ],
    },
    "Septoria_leaf_spot": {
        "title": "برنامج حيوي داعم لتبقع السبتوريا",
        "steps": [
            "إزالة الأوراق السفلية المصابة.",
            "تقليل الرش العلوي الذي يبلل الأوراق.",
            "تحسين التهوية داخل النبات.",
            "المتابعة أسبوعيًا.",
        ],
        "materials": [
            {"type": "Biological", "name": "Bacillus subtilis-based product", "dose_note": "اتبع الملصق."},
        ],
    },
    "Late_blight": {
        "title": "برنامج حيوي داعم للّفحة المتأخرة",
        "steps": [
            "التدخل السريع وإزالة الأوراق ذات الإصابة الشديدة.",
            "خفض الرطوبة وتحسين التهوية.",
            "المتابعة اليومية عند الظروف الملائمة.",
        ],
        "materials": [
            {"type": "Biological", "name": "Bacillus-based protective program", "dose_note": "اتبع الملصق."},
        ],
    },
    "healthy": {
        "title": "برنامج وقائي",
        "steps": [
            "المتابعة الدورية.",
            "الحفاظ على تهوية جيدة.",
            "تجنب الرطوبة الزائدة.",
        ],
        "materials": [],
    },
    "unknown": {
        "title": "برنامج أولي",
        "steps": [
            "إعادة التصوير بصورة أوضح.",
            "فحص الحقل ميدانيًا.",
            "عدم استخدام مبيدات قبل تأكيد التشخيص.",
        ],
        "materials": [],
    },
}

CHEMICAL_PROGRAMS = {
    "Early_blight": {
        "title": "برنامج كيميائي مرجعي للّفحة المبكرة",
        "warning": "استخدم فقط المنتج المسجل محليًا واتبع الملصق.",
        "products": [
            {
                "active_ingredient": "Mancozeb",
                "trade_name_example": "مثال وقائي",
                "dose_template": "حسب الملصق",
                "interval_days": "7–10 أيام",
                "note": "ضمن برنامج تناوب",
            },
            {
                "active_ingredient": "Chlorothalonil",
                "trade_name_example": "مثال وقائي",
                "dose_template": "حسب الملصق",
                "interval_days": "7–10 أيام",
                "note": "تأكد من التسجيل",
            },
        ],
    },
    "Septoria_leaf_spot": {
        "title": "برنامج كيميائي مرجعي لتبقع السبتوريا",
        "warning": "الجرعة النهائية تعتمد على المنتج المسجل والملصق المحلي.",
        "products": [
            {
                "active_ingredient": "Mancozeb",
                "trade_name_example": "مثال وقائي",
                "dose_template": "حسب الملصق",
                "interval_days": "7–10 أيام",
                "note": "مفيد في البداية",
            },
            {
                "active_ingredient": "Copper-based fungicide",
                "trade_name_example": "مثال نحاسي",
                "dose_template": "حسب الملصق",
                "interval_days": "وفق الملصق",
                "note": "تحقق من ملاءمة المحصول",
            },
        ],
    },
    "Late_blight": {
        "title": "برنامج كيميائي مرجعي للّفحة المتأخرة",
        "warning": "اتبع الملصق المحلي فقط.",
        "products": [
            {
                "active_ingredient": "Cymoxanil + Mancozeb",
                "trade_name_example": "مثال خليط",
                "dose_template": "حسب الملصق",
                "interval_days": "وفق الملصق",
                "note": "ضمن برنامج تناوب",
            },
            {
                "active_ingredient": "Metalaxyl-M mixtures",
                "trade_name_example": "مثال مجموعة متخصصة",
                "dose_template": "حسب الملصق",
                "interval_days": "وفق الملصق",
                "note": "راقب المقاومة",
            },
        ],
    },
    "healthy": {
        "title": "لا حاجة لبرنامج علاجي",
        "warning": "لا تستخدم مبيدًا دون حاجة فعلية.",
        "products": [],
    },
    "unknown": {
        "title": "لا توجد توصية كيميائية نهائية",
        "warning": "لا تستخدم أي مبيد قبل تأكيد التشخيص.",
        "products": [],
    },
}

# =========================================================
# أدوات مساعدة
# =========================================================
def serve_page(path: str, not_found_message: str):
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"status": "error", "message": not_found_message})


def read_json_list(path: str, default_list: Optional[List[str]] = None) -> List[str]:
    if not os.path.exists(path):
        return default_list[:] if default_list else []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        tensor_count = sum(1 for v in checkpoint.values() if torch.is_tensor(v))
        if tensor_count > 0:
            return checkpoint

    raise ValueError("Could not extract a valid state_dict from checkpoint.")


def normalize_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    state_dict = strip_prefix_if_present(state_dict, "module.")
    state_dict = strip_prefix_if_present(state_dict, "model.")
    return state_dict


def detect_num_classes_from_state_dict(state_dict: Dict[str, Any], fallback_num_classes: int) -> int:
    candidate_keys = ["fc.weight", "classifier.weight", "classifier.6.weight", "head.weight"]
    for key in candidate_keys:
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return int(state_dict[key].shape[0])

    for key, value in state_dict.items():
        if torch.is_tensor(value) and value.ndim == 2 and (
            key.endswith("fc.weight") or key.endswith("classifier.weight") or key.endswith("head.weight")
        ):
            return int(value.shape[0])

    return fallback_num_classes


def load_resnet18_auto(model_path: str, classes_list: List[str]) -> Tuple[nn.Module, int, List[str]]:
    raw_checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = extract_state_dict(raw_checkpoint)
    state_dict = normalize_state_dict_keys(state_dict)

    num_classes = detect_num_classes_from_state_dict(state_dict, len(classes_list))

    if len(classes_list) < num_classes:
        for i in range(len(classes_list), num_classes):
            classes_list.append(f"Unknown_Class_{i}")
    classes_list = classes_list[:num_classes]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print(f"Loaded: {model_path}")
    print("num_classes =", num_classes)
    print("missing_keys =", missing_keys)
    print("unexpected_keys =", unexpected_keys)

    model.eval()
    return model, num_classes, classes_list


def safe_json_load(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_json_save(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def risk_score_from_log(item: Dict[str, Any]) -> int:
    explicit = item.get("risk_score")
    if explicit is not None:
        try:
            return int(float(explicit))
        except Exception:
            pass

    spread = item.get("spread_risk", "غير متاح")
    decision = item.get("decision", "")
    confidence = float(item.get("confidence", 0) or 0)
    severity_percent = float(item.get("severity_percent", 0) or 0)

    score = 0

    if spread == "مرتفع":
        score += 45
    elif spread == "متوسط":
        score += 28
    elif spread == "منخفض":
        score += 10
    else:
        score += 5

    if decision == "غير مؤكد":
        score += 10
    else:
        score += 5

    if confidence < 40:
        score += 10
    elif confidence < 70:
        score += 6
    else:
        score += 2

    if severity_percent >= 25:
        score += 25
    elif severity_percent >= 10:
        score += 16
    elif severity_percent > 0:
        score += 8

    return min(100, int(round(score)))


def normalize_log_item(item: Dict[str, Any]) -> Dict[str, Any]:
    x = dict(item)

    if "region" not in x or not x.get("region"):
        x["region"] = "غير محددة"

    if "severity_percent" not in x:
        try:
            x["severity_percent"] = float(x.get("severity", {}).get("percent", 0))
        except Exception:
            x["severity_percent"] = 0

    if "severity_label" not in x and isinstance(x.get("severity"), dict):
        x["severity_label"] = x["severity"].get("label", "غير متاح")

    x["risk_score"] = risk_score_from_log(x)
    return x

# =========================================================
# تحميل الكلاسات والموديلات
# =========================================================
PLANT_CLASSES = read_json_list(PLANT_CLASSES_PATH, DEFAULT_PLANT_CLASSES)
GENERAL_CLASSES = read_json_list(GENERAL_CLASSES_PATH, [])
TOMATO_CLASSES = read_json_list(TOMATO_CLASSES_PATH, [])

DEVICE = torch.device("cpu")

plant_model, PLANT_NUM_CLASSES, PLANT_CLASSES = load_resnet18_auto(PLANT_MODEL_PATH, PLANT_CLASSES)
general_model, GENERAL_NUM_CLASSES, GENERAL_CLASSES = load_resnet18_auto(GENERAL_MODEL_PATH, GENERAL_CLASSES)
tomato_model, TOMATO_NUM_CLASSES, TOMATO_CLASSES = load_resnet18_auto(TOMATO_MODEL_PATH, TOMATO_CLASSES)

plant_model.to(DEVICE)
general_model.to(DEVICE)
tomato_model.to(DEVICE)

# =========================================================
# التحويل
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================================================
# مساعدات الأسماء
# =========================================================
PLANT_PREFIXES = [p for p in ARABIC_PLANT_NAMES.keys() if p != "unknown"]


def clean_label(label: str) -> str:
    label = label.replace("___", "_").replace(",", "").replace("(", "").replace(")", "").replace(" ", "_")
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_")


def parse_general_disease_class_name(class_name: str) -> Tuple[str, str, str, str]:
    label = clean_label(class_name)

    detected_plant = None
    for prefix in PLANT_PREFIXES:
        if label.startswith(prefix + "_") or label == prefix:
            detected_plant = prefix
            break

    if detected_plant is None:
        return "unknown", "unknown", "غير معروف", "غير معروف"

    disease_part = label[len(detected_plant):].strip("_")
    disease_en = "healthy" if disease_part == "" or disease_part.lower() == "healthy" else disease_part

    return (
        detected_plant,
        disease_en,
        ARABIC_PLANT_NAMES.get(detected_plant, detected_plant),
        ARABIC_DISEASE_NAMES.get(disease_en, disease_en.replace("_", " "))
    )


def parse_tomato_class_name(class_name: str) -> Tuple[str, str, str, str]:
    label = clean_label(class_name)

    if label.startswith("Tomato_"):
        disease_part = label[len("Tomato_"):].strip("_")
    else:
        disease_part = label

    disease_en = "healthy" if disease_part == "" or disease_part.lower() == "healthy" else disease_part

    return (
        "Tomato",
        disease_en,
        "طماطم",
        ARABIC_DISEASE_NAMES.get(disease_en, disease_en.replace("_", " "))
    )

# =========================================================
# جودة الصورة وشدة الإصابة
# =========================================================
def analyze_image_quality(pil_img: Image.Image) -> Dict[str, Any]:
    rgb = np.array(pil_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(gray))
    warnings = []

    if w < 220 or h < 220:
        warnings.append("الصورة صغيرة الدقة")
    if blur_score < 80:
        warnings.append("الصورة غير واضحة أو مهزوزة")
    if brightness < 50:
        warnings.append("الصورة مظلمة")
    elif brightness > 220:
        warnings.append("الصورة شديدة السطوع")

    quality_status = "جيدة"
    if len(warnings) == 1:
        quality_status = "متوسطة"
    elif len(warnings) >= 2:
        quality_status = "ضعيفة"

    return {
        "width": int(w),
        "height": int(h),
        "blur_score": round(float(blur_score), 2),
        "brightness": round(brightness, 2),
        "quality_status": quality_status,
        "warnings": warnings,
    }


def estimate_severity(pil_img: Image.Image, disease: str) -> Dict[str, Any]:
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower = np.array([5, 40, 20])
    upper = np.array([35, 255, 255])

    lesion_mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_OPEN, kernel)
    lesion_mask = cv2.morphologyEx(lesion_mask, cv2.MORPH_CLOSE, kernel)

    lesion_ratio = float(np.sum(lesion_mask > 0)) / float(lesion_mask.size)
    percent = round(lesion_ratio * 100, 2)

    if disease == "healthy":
        percent = min(percent, 2.0)

    if percent < 10:
        label = "منخفضة"
    elif percent < 25:
        label = "متوسطة"
    else:
        label = "عالية"

    return {"percent": percent, "label": label}

# =========================================================
# Bullseye
# =========================================================
def detect_bullseye_pattern(pil_img: Image.Image) -> Dict[str, Any]:
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 60, 140)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=18,
        param1=80,
        param2=18,
        minRadius=6,
        maxRadius=42,
    )

    edge_density = float(np.mean(edges > 0))
    score = 0.0

    if circles is not None:
        score += min(len(circles[0]) * 0.08, 0.5)
    if 0.08 <= edge_density <= 0.28:
        score += 0.25

    return {
        "detected": bool(score >= 0.30),
        "score": round(min(score, 1.0), 2),
        "edge_density": round(edge_density, 3),
    }

# =========================================================
# التوصيات والبرامج
# =========================================================
def get_general_recommendation(disease: str) -> str:
    d = disease.lower()
    if d == "healthy":
        return "النبات سليم مبدئيًا. استمر في المتابعة والوقاية الجيدة."
    if "late_blight" in d:
        return "التدخل السريع مهم. اخفض الرطوبة وابدأ برنامج حماية مناسب وراقب الانتشار يوميًا."
    if "early_blight" in d:
        return "أزل الأوراق الشديدة الإصابة، وحسّن التهوية، وابدأ برنامج حماية متناوب."
    if "septoria" in d:
        return "أزل الأوراق السفلية المصابة، وخفف البلل الورقي، وراقب تطور البقع."
    if "bacterial_spot" in d:
        return "قلل البلل الورقي وتجنب نقل العدوى بين النباتات والأدوات."
    return "أعد التصوير بصورة أوضح إذا كانت النتيجة غير منطقية، وراجع الحالة ميدانيًا."


def get_bio_program(disease: str) -> Dict[str, Any]:
    return BIO_PROGRAMS.get(disease, BIO_PROGRAMS["unknown"])


def get_chemical_program(disease: str) -> Dict[str, Any]:
    return CHEMICAL_PROGRAMS.get(disease, CHEMICAL_PROGRAMS["unknown"])

# =========================================================
# الأسئلة الإلزامية للطماطم
# =========================================================
def generate_follow_up_questions() -> List[Dict[str, Any]]:
    return [
        {"id": "q1_bullseye", "question": "هل البقع تحتوي على حلقات دائرية متداخلة تشبه الهدف؟"},
        {"id": "q2_small_many", "question": "هل البقع كثيرة وصغيرة جدًا ومتفرقة؟"},
        {"id": "q3_gray_center", "question": "هل مركز البقعة فاتح أو رمادي مع حواف داكنة؟"},
        {"id": "q4_lower_leaves", "question": "هل الإصابة بدأت من الأوراق السفلية؟"},
    ]


def apply_tomato_questionnaire(initial_choice: Dict[str, Any], top_predictions: List[Dict[str, Any]], answers: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    early_item = next((x for x in top_predictions if x["disease"] == "Early_blight"), None)
    septoria_item = next((x for x in top_predictions if x["disease"] == "Septoria_leaf_spot"), None)

    if not early_item or not septoria_item:
        return initial_choice, "تم الإبقاء على النتيجة الأولية لعدم توفر مرضين للمقارنة."

    def is_yes(v):
        return v in [True, "true", "True", 1, "1", "yes", "Yes", "نعم"]

    early_score = 0
    septoria_score = 0

    if is_yes(answers.get("q1_bullseye")):
        early_score += 3
    if is_yes(answers.get("q2_small_many")):
        septoria_score += 3
    if is_yes(answers.get("q3_gray_center")):
        septoria_score += 2
    if is_yes(answers.get("q4_lower_leaves")):
        early_score += 1
        septoria_score += 1

    if early_score > septoria_score:
        return early_item, f"تم تعديل النتيجة بعد الأسئلة لصالح اللفحة المبكرة. (Early={early_score}, Septoria={septoria_score})"
    if septoria_score > early_score:
        return septoria_item, f"تم تعديل النتيجة بعد الأسئلة لصالح تبقع السبتوريا. (Early={early_score}, Septoria={septoria_score})"
    return initial_choice, f"الإجابات لم تكن حاسمة، فتم الإبقاء على النتيجة الأولية. (Early={early_score}, Septoria={septoria_score})"

# =========================================================
# Stage 3 - فلتر التحقق الذكي
# =========================================================
def validate_prediction(
    plant_name: str,
    disease_name: str,
    confidence: float,
    quality_status: str,
    crop_match: bool,
    bullseye_info: Dict[str, Any],
    top_predictions: List[Dict[str, Any]],
) -> Tuple[str, str]:
    decision = "مؤكد"
    notes = []

    if not crop_match:
        decision = "غير مؤكد"
        notes.append("المرض لا يطابق الأمراض المعتادة لهذا المحصول.")

    if quality_status == "ضعيفة":
        decision = "غير مؤكد"
        notes.append("جودة الصورة ضعيفة.")

    if confidence < 70:
        decision = "غير مؤكد"
        notes.append("الثقة منخفضة.")

    if plant_name == "Tomato":
        early_item = next((x for x in top_predictions if x["disease"] == "Early_blight"), None)
        septoria_item = next((x for x in top_predictions if x["disease"] == "Septoria_leaf_spot"), None)

        if early_item and septoria_item:
            diff = abs(early_item["confidence"] - septoria_item["confidence"])
            if diff < 8:
                decision = "غير مؤكد"
                notes.append("هناك تقارب كبير بين اللفحة المبكرة والسبتوريا.")
            if bullseye_info.get("detected") and disease_name == "Septoria_leaf_spot":
                notes.append("ظهور Bullseye يرجح اللفحة المبكرة بصريًا.")

    if not notes:
        notes.append("النتيجة منطقية ومتوافقة مع المحصول وجودة الصورة.")

    return decision, " ".join(notes)

# =========================================================
# التنبؤات
# =========================================================
def predict_plant_stage(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = plant_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    topk = min(5, len(PLANT_CLASSES))
    top_conf, top_idx = torch.topk(probs, topk)

    preds = []
    for i in range(topk):
        idx = top_idx[i].item()
        plant_name = PLANT_CLASSES[idx] if idx < len(PLANT_CLASSES) else f"Unknown_{idx}"
        preds.append({
            "class_name": plant_name,
            "plant": plant_name,
            "plant_ar": ARABIC_PLANT_NAMES.get(plant_name, plant_name),
            "confidence": round(float(top_conf[i].item() * 100), 2),
        })

    return preds[0], preds, probs


def predict_general_disease_stage(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = general_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    topk = min(5, len(GENERAL_CLASSES))
    top_conf, top_idx = torch.topk(probs, topk)

    preds = []
    for i in range(topk):
        idx = top_idx[i].item()
        class_name = GENERAL_CLASSES[idx] if idx < len(GENERAL_CLASSES) else f"Unknown_{idx}"
        plant_name, disease_name, plant_ar, disease_ar = parse_general_disease_class_name(class_name)

        preds.append({
            "class_name": class_name,
            "plant": plant_name,
            "plant_ar": plant_ar,
            "disease": disease_name,
            "disease_ar": disease_ar,
            "confidence": round(float(top_conf[i].item() * 100), 2),
        })

    return preds[0], preds, probs


def predict_tomato_stage(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = tomato_model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    topk = min(5, len(TOMATO_CLASSES))
    top_conf, top_idx = torch.topk(probs, topk)

    preds = []
    for i in range(topk):
        idx = top_idx[i].item()
        class_name = TOMATO_CLASSES[idx] if idx < len(TOMATO_CLASSES) else f"Unknown_{idx}"
        _, disease_name, _, disease_ar = parse_tomato_class_name(class_name)

        preds.append({
            "class_name": class_name,
            "plant": "Tomato",
            "plant_ar": "طماطم",
            "disease": disease_name,
            "disease_ar": disease_ar,
            "confidence": round(float(top_conf[i].item() * 100), 2),
        })

    return preds[0], preds, probs

# =========================================================
# Grad-CAM
# =========================================================
plant_gradients = None
plant_activations = None
general_gradients = None
general_activations = None
tomato_gradients = None
tomato_activations = None


def _plant_bwd(module, grad_input, grad_output):
    global plant_gradients
    plant_gradients = grad_output[0]


def _plant_fwd(module, input, output):
    global plant_activations
    plant_activations = output


def _general_bwd(module, grad_input, grad_output):
    global general_gradients
    general_gradients = grad_output[0]


def _general_fwd(module, input, output):
    global general_activations
    general_activations = output


def _tomato_bwd(module, grad_input, grad_output):
    global tomato_gradients
    tomato_gradients = grad_output[0]


def _tomato_fwd(module, input, output):
    global tomato_activations
    tomato_activations = output


plant_model.layer4[-1].register_forward_hook(_plant_fwd)
plant_model.layer4[-1].register_full_backward_hook(_plant_bwd)

general_model.layer4[-1].register_forward_hook(_general_fwd)
general_model.layer4[-1].register_full_backward_hook(_general_bwd)

tomato_model.layer4[-1].register_forward_hook(_tomato_fwd)
tomato_model.layer4[-1].register_full_backward_hook(_tomato_bwd)


def _generate_gradcam_common(
    pil_image: Image.Image,
    model: nn.Module,
    class_idx: int,
    gradients_ref: str,
    activations_ref: str
) -> str:
    image_resized = pil_image.resize((224, 224))
    img_np = np.array(image_resized)
    img_tensor = transform(image_resized).unsqueeze(0).to(DEVICE)

    model.zero_grad()
    output = model(img_tensor)
    score = output[0, class_idx]
    score.backward()

    gradients_local = globals()[gradients_ref]
    activations_local = globals()[activations_ref]

    pooled_gradients = torch.mean(gradients_local, dim=[0, 2, 3])
    acts = activations_local[0].detach().clone()

    for i in range(acts.shape[0]):
        acts[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(acts, dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)

    file_name = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(GRADCAM_DIR, file_name)
    cv2.imwrite(save_path, overlay)

    return f"/static/gradcam/{file_name}"


def generate_gradcam_plant(img: Image.Image, idx: int) -> str:
    return _generate_gradcam_common(img, plant_model, idx, "plant_gradients", "plant_activations")


def generate_gradcam_general(img: Image.Image, idx: int) -> str:
    return _generate_gradcam_common(img, general_model, idx, "general_gradients", "general_activations")


def generate_gradcam_tomato(img: Image.Image, idx: int) -> str:
    return _generate_gradcam_common(img, tomato_model, idx, "tomato_gradients", "tomato_activations")

# =========================================================
# الطقس وتوقع الانتشار
# =========================================================
def fetch_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "forecast_days": 1,
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return {"latitude": latitude, "longitude": longitude, "current": data.get("current", {})}


def predict_spread_risk(disease: str, weather: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not weather or "current" not in weather:
        return {"level": "غير متاح", "score": None, "reason": "لم يتم توفير الإحداثيات أو تعذر جلب الطقس."}

    current = weather["current"]
    t = current.get("temperature_2m")
    rh = current.get("relative_humidity_2m")
    p = current.get("precipitation")
    wind = current.get("wind_speed_10m")

    score = 0

    if rh is not None:
        if rh >= 85:
            score += 35
        elif rh >= 70:
            score += 20

    if p is not None:
        if p >= 1:
            score += 25
        elif p > 0:
            score += 10

    if wind is not None and wind >= 20:
        score += 8

    d = disease.lower()
    if t is not None:
        if "early_blight" in d and 20 <= t <= 30:
            score += 20
        elif "septoria" in d and 18 <= t <= 27:
            score += 20
        elif "late_blight" in d and 10 <= t <= 24:
            score += 25
        elif 18 <= t <= 30:
            score += 10

    if score >= 70:
        level = "مرتفع"
    elif score >= 40:
        level = "متوسط"
    else:
        level = "منخفض"

    return {
        "level": level,
        "score": score,
        "reason": "تقدير مبني على الرطوبة والحرارة والهطول والرياح.",
    }

# =========================================================
# السجل
# =========================================================
def save_log(entry: Dict[str, Any]) -> None:
    data = safe_json_load(LOG_FILE, [])
    data.append(entry)
    safe_json_save(LOG_FILE, data)


def load_logs() -> List[Dict[str, Any]]:
    data = safe_json_load(LOG_FILE, [])
    return [normalize_log_item(x) for x in data]

# =========================================================
# AUTH
# =========================================================
def load_users():
    return safe_json_load(USERS_FILE, [])


def save_users(users):
    safe_json_save(USERS_FILE, users)


def load_sessions():
    return safe_json_load(SESSIONS_FILE, {})


def save_sessions(sessions):
    safe_json_save(SESSIONS_FILE, sessions)


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100000
    ).hex()


def find_user_by_email(email: str):
    users = load_users()
    email = email.strip().lower()
    for user in users:
        if user.get("email", "").lower() == email:
            return user
    return None


def create_session(user_email: str):
    sessions = load_sessions()
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "email": user_email,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_sessions(sessions)
    return token


def get_current_user_from_token(authorization: Optional[str]):
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.replace("Bearer ", "").strip()
    sessions = load_sessions()
    session = sessions.get(token)
    if not session:
        return None

    email = session.get("email")
    if not email:
        return None

    return find_user_by_email(email)

# =========================================================
# صفحات
# =========================================================
@app.get("/")
def home():
    return serve_page(INDEX_PATH, "index.html not found")


@app.get("/ui")
def ui():
    return serve_page(INDEX_PATH, "index.html not found")


@app.get("/dashboard")
def dashboard_page():
    return serve_page(DASHBOARD_PATH, "dashboard.html not found")


@app.get("/national_center")
def national_center_page():
    return serve_page(NATIONAL_CENTER_PATH, "national_center.html not found")


@app.get("/alerts_page")
def alerts_page():
    return serve_page(ALERTS_PATH, "alerts.html not found")


@app.get("/forecast")
def forecast_page():
    return serve_page(FORECAST_PATH, "forecast.html not found")


@app.get("/forecast_ai_page")
def forecast_ai_page():
    return serve_page(FORECAST_AI_PATH, "forecast_ai.html not found")


@app.get("/admin")
def admin_page():
    return serve_page(ADMIN_PATH, "admin.html not found")


@app.get("/ops")
def ops_page():
    return serve_page(OPS_PATH, "ops.html not found")


@app.get("/map")
def map_page():
    return serve_page(MAP_PATH, "map.html not found")


@app.get("/reports_center")
def reports_center_page():
    return serve_page(REPORTS_CENTER_PATH, "reports_center.html not found")


@app.get("/layout_shell")
def layout_shell_page():
    return serve_page(LAYOUT_SHELL_PATH, "layout_shell.html not found")


@app.get("/login")
def login_page():
    return serve_page(LOGIN_PATH, "login.html not found")


@app.get("/register")
def register_page():
    return serve_page(REGISTER_PATH, "register.html not found")


@app.get("/farmers")
def farmers_page():
    return serve_page(FARMERS_PATH, "farmers.html not found")

COMMAND_CENTER_PATH = os.path.join(FRONTEND_DIR, "command_center.html")

@app.get("/command_center")
def command_center_page():
    return serve_page(COMMAND_CENTER_PATH, "command_center.html not found")

# =========================================================
# API أساسية
# =========================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "plant_model_path": PLANT_MODEL_PATH,
        "plant_num_classes": PLANT_NUM_CLASSES,
        "general_model_path": GENERAL_MODEL_PATH,
        "general_num_classes": GENERAL_NUM_CLASSES,
        "tomato_model_path": TOMATO_MODEL_PATH,
        "tomato_num_classes": TOMATO_NUM_CLASSES,
    }


@app.get("/stats")
def stats():
    logs = load_logs()
    total = len(logs)
    confirmed = sum(1 for x in logs if x.get("decision") == "مؤكد")
    uncertain = sum(1 for x in logs if x.get("decision") == "غير مؤكد")

    plants = Counter(x.get("plant", "Unknown") for x in logs)
    diseases = Counter(x.get("disease", "Unknown") for x in logs)
    systems = Counter(x.get("system_used", "Unknown") for x in logs)

    return {
        "total_predictions": total,
        "confirmed": confirmed,
        "uncertain": uncertain,
        "top_plants": [{"plant": k, "count": v} for k, v in plants.most_common(5)],
        "top_diseases": [{"disease": k, "count": v} for k, v in diseases.most_common(5)],
        "systems_used": [{"system": k, "count": v} for k, v in systems.most_common()],
    }


@app.get("/recent")
def recent():
    logs = load_logs()
    return {"items": list(reversed(logs[-50:]))}


@app.get("/alerts")
def alerts():
    logs = load_logs()
    flagged = [
        x for x in reversed(logs)
        if x.get("decision") == "غير مؤكد" or x.get("spread_risk") == "مرتفع" or not x.get("crop_disease_match", True)
    ]
    return {"items": flagged[:50]}


@app.get("/zones_summary")
def zones_summary():
    logs = load_logs()

    grouped = defaultdict(list)
    for x in logs:
        grouped[x.get("region", "غير محددة")].append(x)

    items = []
    for region, rows in grouped.items():
        count = len(rows)
        confirmed = sum(1 for r in rows if r.get("decision") == "مؤكد")
        avg_risk = round(sum(r.get("risk_score", 0) for r in rows) / count, 2) if count else 0
        avg_severity_percent = round(sum(float(r.get("severity_percent", 0) or 0) for r in rows) / count, 2) if count else 0
        top_disease = Counter(r.get("disease_ar", r.get("disease", "-")) for r in rows).most_common(1)[0][0] if rows else "-"

        items.append({
            "region": region,
            "count": count,
            "confirmed": confirmed,
            "avg_risk": avg_risk,
            "avg_severity_percent": avg_severity_percent,
            "top_disease": top_disease,
        })

    items.sort(key=lambda x: x["avg_risk"], reverse=True)
    return {"items": items}


# مهم: هذا API JSON فقط
@app.get("/forecast_ai")
def forecast_ai_api():
    zones = zones_summary()["items"]

    results = []
    for z in zones:
        base = float(z.get("avg_risk", 0))

        risk_24h = round(min(100, base * 1.10), 1)
        risk_48h = round(min(100, base * 1.25), 1)
        risk_72h = round(min(100, base * 1.40), 1)

        if risk_72h >= 80:
            trend = "تصاعدي مرتفع"
        elif risk_72h >= 45:
            trend = "متوسط"
        else:
            trend = "منخفض"

        results.append({
            "region": z.get("region", "غير محددة"),
            "top_disease": z.get("top_disease", "-"),
            "risk_24h": risk_24h,
            "risk_48h": risk_48h,
            "risk_72h": risk_72h,
            "trend": trend
        })

    results.sort(key=lambda x: x["risk_72h"], reverse=True)
    return {"items": results}


@app.get("/map_data")
def map_data():
    logs = load_logs()
    items = [
        x for x in logs
        if x.get("latitude") is not None and x.get("longitude") is not None
    ]
    return {"items": items}


@app.get("/weather_points")
def weather_points():
    logs = load_logs()
    items = []
    for x in logs:
        weather = x.get("weather")
        current = weather.get("current", {}) if isinstance(weather, dict) else {}
        if x.get("latitude") is not None and x.get("longitude") is not None:
            items.append({
                "latitude": x.get("latitude"),
                "longitude": x.get("longitude"),
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "precipitation": current.get("precipitation"),
                "wind_speed": current.get("wind_speed_10m"),
                "plant_ar": x.get("plant_ar"),
                "disease_ar": x.get("disease_ar"),
            })
    return {"items": items}


@app.get("/regions_geojson")
def regions_geojson():
    geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "غير محددة"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[34, 16], [56, 16], [56, 33], [34, 33], [34, 16]]]
                }
            }
        ]
    }
    return geo

# =========================================================
# Endpoint الرئيسي
# =========================================================
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    plant_name: str = Form("auto"),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    questionnaire_answers: Optional[str] = Form(None),
    region: Optional[str] = Form(None),
):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        selected_plant = (plant_name or "auto").strip()

        image_quality = analyze_image_quality(image)
        bullseye = detect_bullseye_pattern(image)

        if selected_plant.lower() != "auto":
            detected_plant = selected_plant
            detected_plant_ar = ARABIC_PLANT_NAMES.get(detected_plant, detected_plant)
            plant_confidence = 100.0
            plant_top_predictions = [{
                "class_name": detected_plant,
                "plant": detected_plant,
                "plant_ar": detected_plant_ar,
                "confidence": 100.0,
            }]
            plant_gradcam_url = None
            stage1_note = f"تم اعتماد نوع النبات من اختيار المستخدم: {detected_plant_ar}."
        else:
            plant_best, plant_top_predictions, _ = predict_plant_stage(image)
            detected_plant = plant_best["plant"]
            detected_plant_ar = plant_best["plant_ar"]
            plant_confidence = float(plant_best["confidence"])

            plant_idx = PLANT_CLASSES.index(plant_best["class_name"]) if plant_best["class_name"] in PLANT_CLASSES else 0
            plant_gradcam_url = generate_gradcam_plant(image, plant_idx)
            stage1_note = f"تم تحديد النبات تلقائيًا كـ {detected_plant_ar}."

        system_used = ""
        disease_top_predictions: List[Dict[str, Any]] = []
        follow_up_questions: List[Dict[str, Any]] = []
        needs_questions = False

        if detected_plant == "Tomato":
            tomato_best, tomato_predictions, _ = predict_tomato_stage(image)
            chosen = tomato_best
            stage2_note = "تم استخدام موديل طماطم متخصص."

            if bullseye["detected"]:
                early_item = next((x for x in tomato_predictions if x["disease"] == "Early_blight"), None)
                if early_item:
                    chosen = early_item
                    stage2_note += " تم ترجيح اللفحة المبكرة بسبب Bullseye."

            if questionnaire_answers:
                try:
                    answers = json.loads(questionnaire_answers)
                    chosen, q_note = apply_tomato_questionnaire(chosen, tomato_predictions, answers)
                    stage2_note += " " + q_note
                except Exception:
                    stage2_note += " تعذر قراءة إجابات الأسئلة."

            final_plant = "Tomato"
            final_plant_ar = "طماطم"
            final_disease = chosen["disease"]
            final_disease_ar = chosen["disease_ar"]
            final_confidence = float(chosen["confidence"])
            final_class_name = chosen["class_name"]

            disease_top_predictions = tomato_predictions
            follow_up_questions = generate_follow_up_questions()
            needs_questions = True
            system_used = "tomato_ai"

            disease_idx = TOMATO_CLASSES.index(final_class_name) if final_class_name in TOMATO_CLASSES else 0
            disease_gradcam_url = generate_gradcam_tomato(image, disease_idx)

        else:
            general_best, general_predictions, _ = predict_general_disease_stage(image)
            same_crop_predictions = [x for x in general_predictions if x["plant"] == detected_plant]
            chosen = same_crop_predictions[0] if same_crop_predictions else general_best

            final_plant = detected_plant
            final_plant_ar = detected_plant_ar
            final_disease = chosen["disease"]
            final_disease_ar = chosen["disease_ar"]
            final_confidence = float(chosen["confidence"])
            final_class_name = chosen["class_name"]

            disease_top_predictions = same_crop_predictions if same_crop_predictions else general_predictions
            follow_up_questions = []
            needs_questions = False
            system_used = "general_ai"
            stage2_note = "تم استخدام النظام العام للأمراض."

            disease_idx = GENERAL_CLASSES.index(final_class_name) if final_class_name in GENERAL_CLASSES else 0
            disease_gradcam_url = generate_gradcam_general(image, disease_idx)

        allowed_diseases = CROP_DISEASE_MAP.get(final_plant, None)
        crop_match = True
        if allowed_diseases is not None and final_disease not in allowed_diseases:
            crop_match = False

        decision, validation_note = validate_prediction(
            final_plant,
            final_disease,
            final_confidence,
            image_quality["quality_status"],
            crop_match,
            bullseye,
            disease_top_predictions,
        )

        decision_note = f"{stage1_note} {stage2_note} {validation_note}".strip()

        sev = estimate_severity(image, final_disease if final_disease != "unknown" else "healthy")
        recommendation = get_general_recommendation(final_disease)
        bio_program = get_bio_program(final_disease)
        chemical_program = get_chemical_program(final_disease)

        weather_data = None
        spread_prediction = {"level": "غير متاح", "score": None, "reason": "لم يتم تمرير الإحداثيات."}
        if latitude is not None and longitude is not None:
            try:
                weather_data = fetch_weather(latitude, longitude)
                spread_prediction = predict_spread_risk(final_disease, weather_data)
            except Exception as e:
                weather_data = {"error": str(e), "latitude": latitude, "longitude": longitude}

        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "plant": final_plant,
            "plant_ar": final_plant_ar,
            "disease": final_disease,
            "disease_ar": final_disease_ar,
            "confidence": round(final_confidence, 2),
            "decision": decision,
            "system_used": system_used,
            "severity_label": sev["label"],
            "severity_percent": sev["percent"],
            "spread_risk": spread_prediction["level"],
            "crop_disease_match": crop_match,
            "latitude": latitude,
            "longitude": longitude,
            "weather": weather_data,
            "region": (region or "غير محددة").strip() if region is not None else "غير محددة",
        }
        log_entry["risk_score"] = risk_score_from_log(log_entry)
        save_log(log_entry)

        return {
            "status": "ok",
            "system_used": system_used,
            "plant_stage": {
                "detected_plant": detected_plant,
                "detected_plant_ar": detected_plant_ar,
                "confidence": round(plant_confidence, 2),
                "top_predictions": plant_top_predictions,
                "gradcam_url": plant_gradcam_url,
            },
            "final_diagnosis": {
                "plant": final_plant,
                "plant_ar": final_plant_ar,
                "disease": final_disease,
                "disease_ar": final_disease_ar,
                "confidence": round(final_confidence, 2),
                "decision": decision,
                "decision_note": decision_note,
            },
            "crop_disease_match": crop_match,
            "allowed_diseases_for_crop": allowed_diseases,
            "image_quality": image_quality,
            "severity": sev,
            "bullseye_analysis": bullseye,
            "gradcam": {
                "plant_stage_gradcam_url": plant_gradcam_url,
                "disease_stage_gradcam_url": disease_gradcam_url,
            },
            "recommendation": recommendation,
            "bio_program": bio_program,
            "chemical_program": chemical_program,
            "weather": weather_data,
            "spread_prediction": spread_prediction,
            "needs_questions": needs_questions,
            "follow_up_questions": follow_up_questions,
            "top_predictions_general": disease_top_predictions,
            "top_predictions_same_plant": disease_top_predictions,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.post("/predict-plant")
async def predict_plant_compat(
    file: UploadFile = File(...),
    plant_name: str = Form("auto"),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    questionnaire_answers: Optional[str] = Form(None),
    region: Optional[str] = Form(None),
):
    return await predict(
        file=file,
        plant_name=plant_name,
        latitude=latitude,
        longitude=longitude,
        questionnaire_answers=questionnaire_answers,
        region=region,
    )

# =========================================================
# AUTH API
# =========================================================
@app.post("/api/auth/register")
async def register_user(
    full_name: str = Form(...),
    role: str = Form(...),
    email: str = Form(...),
    phone: str = Form(""),
    region: str = Form(""),
    organization: str = Form(""),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    email = email.strip().lower()

    if password != confirm_password:
        return JSONResponse(status_code=400, content={"status": "error", "message": "كلمة المرور وتأكيدها غير متطابقين."})

    if find_user_by_email(email):
        return JSONResponse(status_code=400, content={"status": "error", "message": "هذا البريد مسجل بالفعل."})

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    users = load_users()
    user = {
        "id": uuid.uuid4().hex,
        "full_name": full_name.strip(),
        "role": role.strip(),
        "email": email,
        "phone": phone.strip(),
        "region": region.strip(),
        "organization": organization.strip(),
        "password_hash": password_hash,
        "salt": salt,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_active": True
    }
    users.append(user)
    save_users(users)

    token = create_session(email)

    return {
        "status": "ok",
        "message": "تم إنشاء الحساب بنجاح.",
        "token": token,
        "user": {
            "id": user["id"],
            "full_name": user["full_name"],
            "role": user["role"],
            "email": user["email"],
            "phone": user["phone"],
            "region": user["region"],
            "organization": user["organization"],
        }
    }


@app.post("/api/auth/login")
async def login_user(
    email: str = Form(...),
    password: str = Form(...),
):
    user = find_user_by_email(email)
    if not user:
        return JSONResponse(status_code=401, content={"status": "error", "message": "البريد أو كلمة المرور غير صحيحة."})

    if not user.get("is_active", True):
        return JSONResponse(status_code=403, content={"status": "error", "message": "الحساب غير نشط."})

    computed_hash = hash_password(password, user["salt"])
    if computed_hash != user["password_hash"]:
        return JSONResponse(status_code=401, content={"status": "error", "message": "البريد أو كلمة المرور غير صحيحة."})

    token = create_session(user["email"])

    return {
        "status": "ok",
        "message": "تم تسجيل الدخول بنجاح.",
        "token": token,
        "user": {
            "id": user["id"],
            "full_name": user["full_name"],
            "role": user["role"],
            "email": user["email"],
            "phone": user["phone"],
            "region": user["region"],
            "organization": user["organization"],
        }
    }


@app.get("/api/auth/me")
async def auth_me(authorization: Optional[str] = Header(None)):
    user = get_current_user_from_token(authorization)
    if not user:
        return JSONResponse(status_code=401, content={"status": "error", "message": "غير مصرح."})

    return {
        "status": "ok",
        "user": {
            "id": user["id"],
            "full_name": user["full_name"],
            "role": user["role"],
            "email": user["email"],
            "phone": user["phone"],
            "region": user["region"],
            "organization": user["organization"],
        }
    }


@app.post("/api/auth/logout")
async def logout_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"status": "error", "message": "غير مصرح."})

    token = authorization.replace("Bearer ", "").strip()
    sessions = load_sessions()

    if token in sessions:
        del sessions[token]
        save_sessions(sessions)

    return {"status": "ok", "message": "تم تسجيل الخروج."}

import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
