"""bercak.py: analisis bercak SIMUTEH (dipakai ulang mulai P7)."""
import json
import cv2
import numpy as np
from utils_tepi import mask_daun_awal, bercak_awal

PARAM_BERCAK = {
    "kanal": 2,              # kanal terpilih (2 = R)
    "k": 4.0,                # ambang median + k x MAD
    "se_bentuk": "elips",    # "elips", "persegi", "silang"
    "se_open": 5,            # ukuran SE opening (0 = tidak dipakai)
    "se_close": 5,           # ukuran SE closing (0 = tidak dipakai)
    "luas_min": 20,          # piksel
    "bulat_min": 0.5,
    "rasio_maks": 1.8,
    "batas_ringan": 0.1,     # persen bercak minimum kelas Ringan
    "batas_berat": 2.0,      # persen bercak minimum kelas Berat
    "mm_per_px": 0.4375,
}

_BENTUK = {"elips": cv2.MORPH_ELLIPSE, "persegi": cv2.MORPH_RECT, "silang": cv2.MORPH_CROSS}


def _se(p, uk):
    return cv2.getStructuringElement(_BENTUK[p["se_bentuk"]], (uk, uk))


def bersihkan(biner, p):
    """Opening membuang titik derau dan potongan tipis; closing menutup celah."""
    m = biner
    # TODO 1: opening bila se_open > 0 (membuang titik derau dan tepi/tulang daun yang tipis)
    if p["se_open"] > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _se(p, p["se_open"]))
    # TODO 2: closing bila se_close > 0 (menutup lubang kecil di dalam bercak)
    if p["se_close"] > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _se(p, p["se_close"]))
    return m


def ukur_dan_saring(biner, p):
    """Mengembalikan mask bercak tersaring dan daftar ukuran setiap bercak."""
    keluar, daftar = np.zeros_like(biner), []
    # TODO 3: label komponen terhubung (konektivitas 8)
    n, lbl, st, cen = cv2.connectedComponentsWithStats((biner > 0).astype(np.uint8), connectivity=8)
    mm2 = p["mm_per_px"] ** 2
    for i in range(1, n):
        # TODO 4: ukur luas, keliling, kebulatan, dan rasio aspek
        luas = int(st[i, cv2.CC_STAT_AREA])
        if luas < p["luas_min"]:                 # penyaringan luas lebih dulu (murah)
            continue
        kontur, _ = cv2.findContours((lbl == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
        keliling = cv2.arcLength(kontur[0], True)
        kebulatan = 0.0 if keliling == 0 else 4 * np.pi * luas / keliling ** 2
        w, h = cv2.minAreaRect(kontur[0])[1]
        rasio = max(w, h) / max(min(w, h), 1)
        # TODO 5: simpan hanya komponen yang lolos luas_min, bulat_min, dan rasio_maks
        if kebulatan >= p["bulat_min"] and rasio <= p["rasio_maks"]:
            keluar[lbl == i] = 255
            daftar.append({"label": i, "luas_px": luas, "luas_mm2": round(luas * mm2, 2),
                           "keliling_px": round(float(keliling), 1), "kebulatan": round(float(kebulatan), 2),
                           "rasio_aspek": round(float(rasio), 2),
                           "cx": int(cen[i][0]), "cy": int(cen[i][1])})
    return keluar, daftar


def analisis_daun(img_praproses, p=None):
    """Masukan: citra BGR hasil praproses. Keluaran: kamus hasil analisis."""
    p = {**PARAM_BERCAK, **(p or {})}
    m_daun = mask_daun_awal(img_praproses)
    biner = bercak_awal(img_praproses, m_daun, kanal=p["kanal"], k=p["k"])
    m_bercak, daftar = ukur_dan_saring(bersihkan(biner, p), p)
    # TODO 6: persen bercak = luas bercak / luas daun x 100
    luas_daun = int((m_daun > 0).sum())
    persen = 100.0 * int((m_bercak > 0).sum()) / luas_daun if luas_daun > 0 else 0.0
    # TODO 7: kelas dari batas_ringan dan batas_berat
    if persen >= p["batas_berat"]:
        kelas = "Bercak Berat"
    elif persen >= p["batas_ringan"]:
        kelas = "Bercak Ringan"
    else:
        kelas = "Sehat"
    return {"mask_daun": m_daun, "mask_bercak": m_bercak, "jumlah_bercak": len(daftar),
            "persen_bercak": round(float(persen), 3), "kelas": kelas, "bercak": daftar}


def simpan_param(p, path="../hasil/param_bercak.json"):
    with open(path, "w") as f:
        json.dump({**PARAM_BERCAK, **p}, f, indent=2)
