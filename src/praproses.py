"""praproses.py: rangkaian praproses SIMUTEH (dipakai ulang mulai P5)."""
import json
import cv2
import numpy as np

PARAM_BAWAAN = {
    "lebar": 640,
    "gamma": 1.0,
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "penapis": "median",
    "kernel": 5,          # Menggunakan median 5 berdasarkan hasil yang biasanya bagus
    "sigma": 1.0,
    "tajam_k": 0.0,
}

def praproses(img_bgr, p=None):
    """Urutan: ubah ukuran -> gamma -> CLAHE kanal V -> redam derau -> tajamkan."""
    p = {**PARAM_BAWAAN, **(p or {})}
    
    # TODO 1: ubah ukuran
    h, w = img_bgr.shape[:2]
    skala = p["lebar"] / w
    dimensi = (p["lebar"], int(h * skala))
    img = cv2.resize(img_bgr, dimensi, interpolation=cv2.INTER_AREA)
    
    # TODO 2: koreksi gamma
    if p["gamma"] != 1.0:
        invGamma = 1.0 / p["gamma"]
        tabel = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, tabel)
        
    # TODO 3: CLAHE kanal V
    if p["clahe_clip"] > 0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=p["clahe_clip"], tileGridSize=(p["clahe_tile"], p["clahe_tile"]))
        v_clahe = clahe.apply(v_ch)
        hsv = cv2.merge([h_ch, s_ch, v_clahe])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
    # TODO 4: redam derau
    if p["penapis"] == "median":
        img = cv2.medianBlur(img, p["kernel"])
    elif p["penapis"] == "gaussian":
        img = cv2.GaussianBlur(img, (p["kernel"], p["kernel"]), p["sigma"])
    elif p["penapis"] == "rerata":
        img = cv2.blur(img, (p["kernel"], p["kernel"]))
        
    # TODO 5: unsharp masking
    if p["tajam_k"] > 0:
        blur = cv2.GaussianBlur(img, (5, 5), 1.0)
        img = cv2.addWeighted(img, 1 + p["tajam_k"], blur, -p["tajam_k"], 0)
        
    return img

def simpan_param(p, path="../hasil/param_praproses.json"):
    with open(path, "w") as f:
        json.dump({**PARAM_BAWAAN, **p}, f, indent=2)
