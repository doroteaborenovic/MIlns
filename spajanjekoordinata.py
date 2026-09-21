import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Putanja do tvoje slike i modela koji već imaš na računaru
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

    # Pravila spajanja tačaka tela (kostur):
    # npr. (11, 13) spaja levo rame sa levim laktom, (13, 15) spaja lakat sa šakom
    LINIJE_SKELETA = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Lice
        (9, 10),                                                         # Usne
        (11, 12),                                                        # Ramena
        (11, 13), (13, 15),                                              # Leva ruka
        (12, 14), (14, 16),                                              # Desna ruka
        (11, 23), (12, 24), (23, 24),                                    # Trup i kukovi
        (23, 25), (25, 27), (27, 29), (29, 31),                          # Leva noga
        (24, 26), (26, 28), (28, 30), (30, 32)                           # Desna noga
    ]

    # 1. Spajamo zglobove linijama
    for pocetak, kraj in LINIJE_SKELETA:
        tacka1 = (int(landmarks[pocetak].x * sirina), int(landmarks[pocetak].y * visina))
        tacka2 = (int(landmarks[kraj].x * sirina), int(landmarks[kraj].y * visina))
        cv2.line(slika, tacka1, tacka2, (255, 100, 0), 3) # Plave linije skeleta

    # 2. Preko linija crtamo zglobove (crvene tačke)
    for lm in landmarks:
        px = int(lm.x * sirina)
        py = int(lm.y * visina)
        cv2.circle(slika, (px, py), 5, (0, 0, 255), -1)

    print("Uspešno: Skelet je formiran spajanjem koordinata!")

    cv2.namedWindow("Korak 2 - Spojeni Skelet", cv2.WINDOW_NORMAL)
    cv2.imshow("Korak 2 - Spojeni Skelet", slika)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detector.close()