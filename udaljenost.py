import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random
import math
import os

PUTANJA_SLIKE = r"C:\Users\PC\gitara\MI2026lns\poza3.jpg"
MODEL_FAJL = "keypoints_detection.task"

# Inicijalizacija detektora
base_options = python.BaseOptions(model_asset_path=MODEL_FAJL)
options = vision.PoseLandmarkerOptions(base_options=base_options, min_pose_detection_confidence=0.6)
detector = vision.PoseLandmarker.create_from_options(options)

mp_image = mp.Image.create_from_file(PUTANJA_SLIKE)
slika = cv2.imread(PUTANJA_SLIKE)
visina, sirina, _ = slika.shape

rezultati = detector.detect(mp_image)

if rezultati.pose_landmarks:
    landmarks = rezultati.pose_landmarks[0]

    # -------------------------------------------------------------
    # KORAK A: Generisanje random mete na slici (Crvena tacka)
    # -------------------------------------------------------------
    meta_x = random.randint(100, sirina - 100)
    meta_y = random.randint(100, visina - 100)

    # -------------------------------------------------------------
    # KORAK B: KAKO DOBIJAMO VREDNOST U METRIMA? (MATEMATIKA)
    # -------------------------------------------------------------
    # 1. Uzimamo nos (tačka 0) i desni gležanj (tačka 28)
    nos = landmarks[0]
    glezanj = landmarks[28]

    # 2. Računamo koliko piksela osoba zauzima po visini na slici
    visina_osobe_pikseli = abs(int(glezanj.y * visina) - int(nos.y * visina))
    if visina_osobe_pikseli == 0:
        visina_osobe_pikseli = visina * 0.7

    # 3. Znamo da je prosecna ljudska visina oko 1.75 metara.
    #    Deljenjem 1.75 metara sa brojem piksela dobijamo vrednost 1 piksela u metrima:
    PROSECNA_VISINA_COVEKA_METRI = 1.75   #visina real osobe u metrima
    METARA_PO_PIKSELU = PROSECNA_VISINA_COVEKA_METRI / visina_osobe_pikseli

    # -------------------------------------------------------------
    # KORAK C: Udaljenost DESNE ŠAKE (Tačka 16 - Just Dance fokus)
    # -------------------------------------------------------------
    saka = landmarks[16]
    saka_x = int(saka.x * sirina)
    saka_y = int(saka.y * visina)

    # Pitagorina teorema za udaljenost u pikselima: c = sqrt(a^2 + b^2)
    udaljenost_saka_px = math.hypot(meta_x - saka_x, meta_y - saka_y)

    # Pretvaranje u metre množenjem sa faktorom razmere:
    udaljenost_saka_metri = udaljenost_saka_px * METARA_PO_PIKSELU

    # -------------------------------------------------------------
    # KORAK D: CRTANJE NA SLICI
    # -------------------------------------------------------------
    # Iscrtavamo osnovni skelet
    LINIJE = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28)]
    for s, e in LINIJE:
        p1 = (int(landmarks[s].x * sirina), int(landmarks[s].y * visina))
        p2 = (int(landmarks[e].x * sirina), int(landmarks[e].y * visina))
        cv2.line(slika, p1, p2, (255, 100, 0), 2)

    # Označavamo šaku zelenim krugom, a metu crvenim
    cv2.circle(slika, (saka_x, saka_y), 10, (0, 255, 0), -1)
    cv2.circle(slika, (meta_x, meta_y), 12, (0, 0, 255), -1)

    # Povezujemo šaku i metu žutom linijom
    cv2.line(slika, (saka_x, saka_y), (meta_x, meta_y), (0, 255, 255), 3)

    # Ispisujemo metre na sredini te linije
    sredina_x = (saka_x + meta_x) // 2
    sredina_y = (saka_y + meta_y) // 2
    cv2.putText(slika, f"{udaljenost_saka_metri:.2f} m", (sredina_x + 10, sredina_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # -------------------------------------------------------------
    # KORAK E: TABELA SA SVIH 33 TAČAKA U TERMINALU
    # -------------------------------------------------------------
    nazivi_tacaka = [
        "Nos", "Unutrasnji ugao levog oka", "Levo oko", "Spoljasnji ugao levog oka",
        "Unutrasnji ugao desnog oka", "Desno oko", "Spoljasnji ugao desnog oka",
        "Levo uvo", "Desno uvo", "Leva ivica usana", "Desna ivica usana",
        "Levo rame", "Desno rame", "Levi lakat", "Desni lakat",
        "Leva saka", "Desna saka (JUST DANCE)", "Levi mali prst", "Desni mali prst",
        "Levi kaziprst", "Desni kaziprst", "Levi palac", "Desni palac",
        "Levi kuk", "Desni kuk", "Levo koleno", "Desno koleno",
        "Levi glezanj", "Desni glezanj", "Leva peta", "Desna peta",
        "Vrh levog stopala", "Vrh desnog stopala"
    ]

    print("\n" + "="*70)
    print(f"POZICIJA SLUČAJNO IZABRANE METE: ({meta_x}, {meta_y})")
    print(f"RAZMERA: 1 piksel na ovoj slici vredi {METARA_PO_PIKSELU * 100:.3f} cm")
    print(f"DESNA ŠAKA (FOKUS) -> Udaljenost: {udaljenost_saka_metri:.2f} metara ({int(udaljenost_saka_px)} px)")
    print("="*70)
    print(f"{'ID':<4} | {'Naziv tacke':<26} | {'X':<5} | {'Y':<5} | {'Udaljenost (m)':<14} | {'(px)'}")
    print("-" * 70)

    for i, lm in enumerate(landmarks):
        lx = int(lm.x * sirina)
        ly = int(lm.y * visina)
        d_px = math.hypot(meta_x - lx, meta_y - ly)
        d_m = d_px * METARA_PO_PIKSELU
        fokus = " <--" if i == 16 else ""
        print(f"{i:<4} | {nazivi_tacaka[i] + fokus:<26} | {lx:<5} | {ly:<5} | {d_m:>6.2f} m        | {d_px:>6.1f} px")
    print("-" * 70)

    cv2.namedWindow("Korak 3 - Merenje Udaljenosti", cv2.WINDOW_NORMAL)
    cv2.imshow("Korak 3 - Merenje Udaljenosti", slika)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detector.close()