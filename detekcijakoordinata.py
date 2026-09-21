import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

PUTANJA_SLIKE = r"C:\Users\PC\gitara\MI2026lns\poza3.jpg"

# preuzimanje ai modela 
MODEL_FAJL = "keypoints_detection.task"
if not os.path.exists(MODEL_FAJL):   #ovo znaci da ako model ne postoji na racunaru, preuzmi ga sa interneta
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, MODEL_FAJL)

# Pokretanje detektora
base_options = python.BaseOptions(model_asset_path=MODEL_FAJL)
options = vision.PoseLandmarkerOptions(base_options=base_options, min_pose_detection_confidence=0.6)
detector = vision.PoseLandmarker.create_from_options(options)

mp_image = mp.Image.create_from_file(PUTANJA_SLIKE)
slika = cv2.imread(PUTANJA_SLIKE)
visina, sirina, _ = slika.shape

rezultati = detector.detect(mp_image)

if rezultati.pose_landmarks:
    landmarks = rezultati.pose_landmarks[0]
    
    print("\n--- DETEKTOVANE KOORDINATE (33 TAČKE) ---")
    # Samo crtamo tačkice (bez spajanja linijama)
    for i, lm in enumerate(landmarks):
        px = int(lm.x * sirina)
        py = int(lm.y * visina)
        
        # Crtamo žutu tačku na svakom zglobu
        cv2.circle(slika, (px, py), 6, (0, 255, 255), -1)
        
        # Ispisujemo prvih nekoliko u terminal da vide
        if i in [0, 11, 12, 15, 16, 23, 24, 27, 28]:
            print(f"Tačka {i:>2}: X = {px}, Y = {py}")

    cv2.namedWindow("Korak 1 - Detekcija Tacaka", cv2.WINDOW_NORMAL)
    cv2.imshow("Korak 1 - Detekcija Tacaka", slika)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detector.close()