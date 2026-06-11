#!/usr/bin/env python3
"""
zamegli_obraz_in_glas.py

Zamegli obraz(e) na videu IN spremeni "barvo" (visino) glasu.
Vse se zgodi lokalno na tvojem racunalniku.

ZAHTEVE (enkratna namestitev):
    pip install opencv-python
    + namescen ffmpeg (Mac: brew install ffmpeg)

UPORABA:
    python3 zamegli_obraz_in_glas.py /Users/jangole/Downloads/IMG_2194.mov

    # globlji glas (privzeto), mocnejsa zameglitev:
    python3 zamegli_obraz_in_glas.py vhod.mov --pitch 0.85 --blur 45

    # visji glas:
    python3 zamegli_obraz_in_glas.py vhod.mov --pitch 1.25

Rezultat: <ime>_anon.mp4 v isti mapi.
"""

import argparse
import os
import subprocess
import sys
import tempfile

try:
    import cv2
except ImportError:
    sys.exit("Manjka opencv. Zazeni: pip install opencv-python")


def find_cascade():
    # Haar kaskada je vgrajena v opencv - brez prenosa.
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(path):
        sys.exit("Ne najdem Haar kaskade v opencv namestitvi.")
    return cv2.CascadeClassifier(path)


def blur_faces(in_path, tmp_video, blur_strength, expand):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        sys.exit(f"Ne morem odpreti videa: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_video, fourcc, fps, (w, h))

    face_cascade = find_cascade()
    # blur_strength mora biti liho stevilo za GaussianBlur
    k = max(3, int(blur_strength) | 1)

    # Detekcijo izvajamo na pomanjsani sliki (mnogo hitreje pri visoki locljivosti),
    # pravokotnike pa nato povecamo nazaj na polno locljivost.
    det_w = 640
    scale = det_w / float(w) if w > det_w else 1.0
    det_min = max(20, int(40 * scale))

    last_boxes = []
    miss = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            small_gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
        else:
            small_gray = gray
        det = face_cascade.detectMultiScale(small_gray, 1.1, 5,
                                            minSize=(det_min, det_min))
        faces = [(int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))
                 for (x, y, fw, fh) in det]

        if len(faces) > 0:
            last_boxes = faces
            miss = 0
        else:
            # ce v tem okvirju ne najde obraza, se nekaj okvirjev drzi zadnjih,
            # da zameglitev ne "utripa"
            miss += 1
            if miss > 8:
                last_boxes = []

        for (x, y, fw, fh) in last_boxes:
            ex = int(fw * expand)
            ey = int(fh * expand)
            x0 = max(0, x - ex)
            y0 = max(0, y - ey)
            x1 = min(w, x + fw + ex)
            y1 = min(h, y + fh + ey)
            roi = frame[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            roi = cv2.GaussianBlur(roi, (k, k), 0)
            # dodaten pikselizacijski sloj za mocnejso anonimizacijo
            small = cv2.resize(roi, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)),
                               interpolation=cv2.INTER_LINEAR)
            roi = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            frame[y0:y1, x0:x1] = roi

        out.write(frame)
        if total and idx % 30 == 0:
            print(f"  obdelano {idx}/{total} okvirjev", end="\r")

    cap.release()
    out.release()
    print(f"\n  zameglitev koncana ({idx} okvirjev)")
    return fps


def change_voice_and_mux(orig_path, blurred_video, out_path, pitch):
    """
    Spremeni visino glasu z asetrate trikom in obdrzi prvotno trajanje
    z atempo kompenzacijo. Nato zdruzi z zamegljeno sliko.
    """
    # asetrate*pitch spremeni visino + hitrost; atempo(1/pitch) povrne hitrost.
    inv = 1.0 / pitch
    # atempo sprejema 0.5-2.0; pri ekstremih razbijemo na vec stopenj
    def atempo_chain(factor):
        parts = []
        while factor < 0.5 or factor > 2.0:
            step = 2.0 if factor > 2.0 else 0.5
            parts.append(step)
            factor /= step
        parts.append(factor)
        return ",".join(f"atempo={p:.4f}" for p in parts)

    af = f"asetrate=44100*{pitch},aresample=44100,{atempo_chain(inv)}"

    cmd = [
        "ffmpeg", "-y",
        "-i", blurred_video,   # 0: zamegljena slika (brez zvoka)
        "-i", orig_path,       # 1: original (za zvok)
        "-filter_complex", f"[1:a]{af}[aout]",
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    print("  spreminjam glas in zdruzujem (ffmpeg)...")
    r = subprocess.run(cmd, stderr=subprocess.PIPE)
    if r.returncode != 0:
        # mogoce original nima zvoka -> samo slika
        print("  opozorilo: ffmpeg z zvokom ni uspel, poskusam brez zvoka.")
        print(r.stderr.decode(errors="ignore")[-800:])
        subprocess.run([
            "ffmpeg", "-y", "-i", blurred_video,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path,
        ], check=True)


def main():
    ap = argparse.ArgumentParser(description="Zamegli obraz in spremeni glas na videu.")
    ap.add_argument("video", help="pot do vhodnega videa (npr. IMG_2194.mov)")
    ap.add_argument("--pitch", type=float, default=0.85,
                    help="visina glasu: <1 globlje, >1 visje (privzeto 0.85)")
    ap.add_argument("--blur", type=int, default=45,
                    help="moc zameglitve (privzeto 45)")
    ap.add_argument("--expand", type=float, default=0.25,
                    help="koliko cez rob obraza naj sega zameglitev (privzeto 0.25)")
    ap.add_argument("-o", "--out", default=None, help="izhodna datoteka")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"Datoteka ne obstaja: {args.video}")

    base, _ = os.path.splitext(args.video)
    out_path = args.out or f"{base}_anon.mp4"

    with tempfile.TemporaryDirectory() as td:
        tmp_video = os.path.join(td, "blurred.mp4")
        print("1/2 Zamegljujem obraz...")
        blur_faces(args.video, tmp_video, args.blur, args.expand)
        print("2/2 Spreminjam glas...")
        change_voice_and_mux(args.video, tmp_video, out_path, args.pitch)

    print(f"\nKONCANO -> {out_path}")


if __name__ == "__main__":
    main()
