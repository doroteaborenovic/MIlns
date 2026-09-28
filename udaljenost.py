# Biblioteka za obradu slika i crtanje po slici
import cv2

# MediaPipe za detekciju ljudskog tela i ključnih tačaka
import mediapipe as mp

# Uvoz osnovnih MediaPipe klasa
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Za generisanje nasumične mete
import random

# Za matematičke funkcije (udaljenost, koren...)
import math

# Rad sa fajlovima i putanjama
import os


# Putanja do slike koju analiziramo
PUTANJA_SLIKE = r"C:\Users\PC\gitara\MI2026lns\poza3.jpg"

# Trenirani model za prepoznavanje poza
MODEL_FAJL = "keypoints_detection.task"


# ==========================================================
# KORAK 1 - Inicijalizacija Pose Landmarker modela
# ==========================================================

# Učitavanje AI modela
base_options = python.BaseOptions(model_asset_path=MODEL_FAJL)

# Podešavanje opcija detektora
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    min_pose_detection_confidence=0.6
)

# Kreiranje detektora poze
detector = vision.PoseLandmarker.create_from_options(options)


# ==========================================================
# KORAK 2 - Učitavanje slike
# ==========================================================

# MediaPipe format slike
mp_image = mp.Image.create_from_file(PUTANJA_SLIKE)

# OpenCV format slike
slika = cv2.imread(PUTANJA_SLIKE)

# Dobijamo dimenzije slike
visina, sirina, _ = slika.shape


# ==========================================================
# KORAK 3 - Pokretanje detekcije tela
# ==========================================================

# Model traži 33 tačke na ljudskom telu
rezultati = detector.detect(mp_image)


# Ako je osoba uspešno pronađena
if rezultati.pose_landmarks:

    # Uzimamo prvu pronađenu osobu
    landmarks = rezultati.pose_landmarks[0]

    # ======================================================
    # KORAK A - Generisanje slučajne mete
    # ======================================================

    # Nasumična X koordinata mete
    meta_x = random.randint(100, sirina - 100)

    # Nasumična Y koordinata mete
    meta_y = random.randint(100, visina - 100)



    # ======================================================
    # KORAK B - Računanje razmere piksel -> metri
    # ======================================================

    # Landmark 0 = nos
    nos = landmarks[0]

    # Landmark 28 = desni gležanj
    glezanj = landmarks[28]

    # Visina osobe izražena u pikselima
    visina_osobe_pikseli = abs(
        int(glezanj.y * visina) -
        int(nos.y * visina)
    )

    # Zaštita od deljenja nulom
    if visina_osobe_pikseli == 0:
        visina_osobe_pikseli = visina * 0.7

    # Pretpostavljena realna visina čoveka
    PROSECNA_VISINA_COVEKA_METRI = 1.75

    # Koliko metara predstavlja jedan piksel
    METARA_PO_PIKSELU = (
        PROSECNA_VISINA_COVEKA_METRI /
        visina_osobe_pikseli
    )



    # ======================================================
    # KORAK C - Udaljenost desne šake od mete
    # ======================================================

    # Landmark 16 = desna šaka
    saka = landmarks[16]

    # Pretvaranje normalizovanih koordinata u piksele
    saka_x = int(saka.x * sirina)
    saka_y = int(saka.y * visina)

    # Euklidska udaljenost između mete i šake
    udaljenost_saka_px = math.hypot(
        meta_x - saka_x,
        meta_y - saka_y
    )

    # Pretvaranje udaljenosti u metre
    udaljenost_saka_metri = (
        udaljenost_saka_px *
        METARA_PO_PIKSELU
    )



    # ======================================================
    # KORAK D - Crtanje skeleta i rezultata
    # ======================================================

    # Parovi tačaka koje povezujemo linijama
    LINIJE = [
        (11, 12),  # ramena
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (25, 27),
        (24, 26),
        (26, 28)
    ]

    # Crtanje skeleta
    for s, e in LINIJE:

        # Prva tačka linije
        p1 = (
            int(landmarks[s].x * sirina),
            int(landmarks[s].y * visina)
        )

        # Druga tačka linije
        p2 = (
            int(landmarks[e].x * sirina),
            int(landmarks[e].y * visina)
        )

        # Linija između tačaka tela
        cv2.line(slika, p1, p2, (255, 100, 0), 2)



    # Zelena tačka = desna šaka
    cv2.circle(
        slika,
        (saka_x, saka_y),
        10,
        (0, 255, 0),
        -1
    )

    # Crvena tačka = meta
    cv2.circle(
        slika,
        (meta_x, meta_y),
        12,
        (0, 0, 255),
        -1
    )

    # Žuta linija između šake i mete
    cv2.line(
        slika,
        (saka_x, saka_y),
        (meta_x, meta_y),
        (0, 255, 255),
        3
    )



    # Računanje sredine linije
    sredina_x = (saka_x + meta_x) // 2
    sredina_y = (saka_y + meta_y) // 2

    # Ispis udaljenosti u metrima
    cv2.putText(
        slika,
        f"{udaljenost_saka_metri:.2f} m",
        (sredina_x + 10, sredina_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )



    # ======================================================
    # KORAK E - Nazivi svih 33 landmark tačke
    # ======================================================

    nazivi_tacaka = [
        "Nos",
        "Unutrasnji ugao levog oka",
        "Levo oko",
        "Spoljasnji ugao levog oka",
        "Unutrasnji ugao desnog oka",
        "Desno oko",
        "Spoljasnji ugao desnog oka",
        "Levo uvo",
        "Desno uvo",
        "Leva ivica usana",
        "Desna ivica usana",
        "Levo rame",
        "Desno rame",
        "Levi lakat",
        "Desni lakat",
        "Leva saka",
        "Desna saka (JUST DANCE)",
        "Levi mali prst",
        "Desni mali prst",
        "Levi kaziprst",
        "Desni kaziprst",
        "Levi palac",
        "Desni palac",
        "Levi kuk",
        "Desni kuk",
        "Levo koleno",
        "Desno koleno",
        "Levi glezanj",
        "Desni glezanj",
        "Leva peta",
        "Desna peta",
        "Vrh levog stopala",
        "Vrh desnog stopala"
    ]


    # Prikaz osnovnih informacija
    print("\n" + "=" * 70)

    print(
        f"POZICIJA META: ({meta_x}, {meta_y})"
    )

    print(
        f"1 piksel = "
        f"{METARA_PO_PIKSELU * 100:.3f} cm"
    )

    print(
        f"DESNA SAKA -> "
        f"{udaljenost_saka_metri:.2f} m "
        f"({int(udaljenost_saka_px)} px)"
    )

    print("=" * 70)


    # Zaglavlje tabele
    print(
        f"{'ID':<4} | "
        f"{'Naziv tacke':<26} | "
        f"{'X':<5} | "
        f"{'Y':<5} | "
        f"{'Udaljenost (m)':<14} | "
        f"{'(px)'}"
    )

    print("-" * 70)



    # Prolazak kroz svih 33 landmarka
    for i, lm in enumerate(landmarks):

        # Piksel koordinate tačke
        lx = int(lm.x * sirina)
        ly = int(lm.y * visina)

        # Udaljenost tačke do mete
        d_px = math.hypot(
            meta_x - lx,
            meta_y - ly
        )

        # Pretvaranje u metre
        d_m = d_px * METARA_PO_PIKSELU

        # Označavanje desne šake
        fokus = " <--" if i == 16 else ""

        # Ispis reda tabele
        print(
            f"{i:<4} | "
            f"{nazivi_tacaka[i] + fokus:<26} | "
            f"{lx:<5} | "
            f"{ly:<5} | "
            f"{d_m:>6.2f} m | "
            f"{d_px:>6.1f} px"
        )

    print("-" * 70)



    # ======================================================
    # KORAK F - Prikaz slike
    # ======================================================

    cv2.namedWindow(
        "Korak 3 - Merenje Udaljenosti",
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        "Korak 3 - Merenje Udaljenosti",
        slika
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Oslobađanje memorije i gašenje modela
detector.close()