# OpenCV biblioteka za rad sa slikama i crtanje po njima
import cv2

# MediaPipe biblioteka za detekciju ljudi i delova tela
import mediapipe as mp

# Uvoz MediaPipe klase za pokretanje AI modela
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Biblioteka za rad sa fajl sistemom
import os


# ==========================================================
# PUTANJE DO FAJLOVA
# ==========================================================

# Putanja do slike koju analiziramo
PUTANJA_SLIKE = r"C:\Users\PC\gitara\MI2026lns\poza3.jpg"

# Naziv MediaPipe modela koji prepoznaje pozu čoveka
MODEL_FAJL = "keypoints_detection.task"


# ==========================================================
# INICIJALIZACIJA MODELA
# ==========================================================

# Učitavanje AI modela sa diska
base_options = python.BaseOptions(
    model_asset_path=MODEL_FAJL
)

# Podešavanje parametara modela
options = vision.PoseLandmarkerOptions(
    base_options=base_options,

    # Potrebno je najmanje 60% sigurnosti da bi poza bila prihvaćena
    min_pose_detection_confidence=0.6
)

# Kreiranje PoseLandmarker objekta
# Ovaj objekat će pronalaziti tačke tela
detector = vision.PoseLandmarker.create_from_options(options)


# ==========================================================
# UČITAVANJE SLIKE
# ==========================================================

# Učitavanje slike u MediaPipe format
mp_image = mp.Image.create_from_file(PUTANJA_SLIKE)

# Učitavanje iste slike u OpenCV format
# OpenCV koristimo za crtanje skeleta
slika = cv2.imread(PUTANJA_SLIKE)

# Uzimamo dimenzije slike
# visina = broj piksela po visini
# sirina = broj piksela po širini
visina, sirina, _ = slika.shape


# ==========================================================
# DETEKCIJA POZE
# ==========================================================

# AI model pokušava da pronađe ljudsko telo
# i svih 33 ključnih tačaka
rezultati = detector.detect(mp_image)


# Ako je pronađena barem jedna osoba
if rezultati.pose_landmarks:

    # Uzimamo prvu detektovanu osobu
    landmarks = rezultati.pose_landmarks[0]

    # ======================================================
    # DEFINISANJE SKELETA
    # ======================================================

    # Svaki par brojeva predstavlja dve tačke
    # koje treba spojiti linijom.
    #
    # Primer:
    # (11,13) = levo rame -> levi lakat
    # (13,15) = levi lakat -> leva šaka
    #
    # Na ovaj način formiramo kompletan skelet.

    LINIJE_SKELETA = [

        # Lice
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),

        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),

        # Usne
        (9, 10),

        # Ramena
        (11, 12),

        # Leva ruka
        (11, 13),
        (13, 15),

        # Desna ruka
        (12, 14),
        (14, 16),

        # Trup i kukovi
        (11, 23),
        (12, 24),
        (23, 24),

        # Leva noga
        (23, 25),
        (25, 27),
        (27, 29),
        (29, 31),

        # Desna noga
        (24, 26),
        (26, 28),
        (28, 30),
        (30, 32)
    ]


    # ======================================================
    # CRTANJE SKELETA
    # ======================================================

    # Prolazak kroz sve parove tačaka
    for pocetak, kraj in LINIJE_SKELETA:

        # Uzimamo koordinatu prve tačke

        tacka1 = (
            int(landmarks[pocetak].x * sirina),
            int(landmarks[pocetak].y * visina)
        )

        # Uzimamo koordinatu druge tačke

        tacka2 = (
            int(landmarks[kraj].x * sirina),
            int(landmarks[kraj].y * visina)
        )

        # Crtanje linije između dve tačke
        # (255,100,0) predstavlja boju linije
        # 3 predstavlja debljinu linije

        cv2.line(
            slika,
            tacka1,
            tacka2,
            (255, 100, 0),
            3
        )


    # ======================================================
    # CRTANJE ZGLOBOVA
    # ======================================================

    # Landmark predstavlja jednu tačku tela
    # npr. rame, lakat, koleno, nos...

    for lm in landmarks:

        # MediaPipe vraća normalizovane koordinate
        # između 0 i 1.

        # Pretvaramo ih u stvarne piksele slike.

        px = int(lm.x * sirina)
        py = int(lm.y * visina)

        # Crtamo crvenu kružnicu na svakoj tački tela
        cv2.circle(
            slika,
            (px, py),
            5,
            (0, 0, 255),
            -1
        )


    # ======================================================
    # INFORMACIJA U TERMINALU
    # ======================================================

    print(
        "Uspešno: Skelet je formiran spajanjem koordinata!"
    )


    # ======================================================
    # PRIKAZ REZULTATA
    # ======================================================

    # Kreiranje OpenCV prozora
    cv2.namedWindow(
        "Korak 2 - Spojeni Skelet",
        cv2.WINDOW_NORMAL
    )

    # Prikaz slike sa iscrtanim skeletom
    cv2.imshow(
        "Korak 2 - Spojeni Skelet",
        slika
    )

    # Čekanje da korisnik pritisne neki taster
    cv2.waitKey(0)

    # Zatvaranje svih OpenCV prozora
    cv2.destroyAllWindows()


# ==========================================================
# GAŠENJE MODELA
# ==========================================================

# Oslobađanje memorije koju koristi MediaPipe model
detector.close()