# ---------------------------------------------------------
# 1. UVOZ POTREBNIH BIBLIOTEKA
# ---------------------------------------------------------
import cv2  # OpenCV biblioteka za obradu slike i prikaz prozora
import mediapipe as mp  # MediaPipe radni okvir za zadatke mašinskog učenja
from mediapipe.tasks import python  # Python API za MediaPipe Tasks
from mediapipe.tasks.python import vision  # Moduli za računarski vid (Pose Landmarker)
import urllib.request  # Biblioteka za preuzimanje fajlova sa interneta
import os  # Biblioteka za rad sa fajl sistemom (provera postojanja fajla)

# ---------------------------------------------------------
# 2. DEFINISANJE PUTANJA I PREUZIMANJE MODELA
# ---------------------------------------------------------
# Putanja do ulazne slike na računaru
PUTANJA_SLIKE = r"C:\Users\PC\gitara\MI2026lns\latinoposeslika.webp"

# Ime pod kojim će se model sačuvati lokalno
MODEL_FAJL = "keypoints_detection.task"

# Provera da li model već postoji na računaru; ako ne postoji, preuzima se sa interneta
if not os.path.exists(MODEL_FAJL):
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
    urllib.request.urlretrieve(url, MODEL_FAJL)

# ---------------------------------------------------------
# 3. KONFIGURACIJA I INICIJALIZACIJA DETEKTORA POZE
# ---------------------------------------------------------
# Postavljanje putanje do preuzetog modela
base_options = python.BaseOptions(model_asset_path=MODEL_FAJL)

# Podešavanje opcija za detektor (minimalni prag pouzdanosti detekcije je 60%)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, 
    min_pose_detection_confidence=0.6
)

# Kreiranje instance detektora poze na osnovu zadatih opcija
detector = vision.PoseLandmarker.create_from_options(options)

# ---------------------------------------------------------
# 4. UČITAVANJE SLIKE
# ---------------------------------------------------------
# Učitavanje slike u MediaPipe formatu (za analizu modelom)
mp_image = mp.Image.create_from_file(PUTANJA_SLIKE)

# Učitavanje iste slike preko OpenCV-a (za crtanje tačaka i prikaz)
slika = cv2.imread(PUTANJA_SLIKE)

# Čitanje dimenzija slike (visina i širina) radi preračunavanja koordinata
visina, sirina, _ = slika.shape

# ---------------------------------------------------------
# 5. DETEKCIJA I OBRADA REZULTATA
# ---------------------------------------------------------
# Pokretanje detekcije ključnih tačaka tela na slici
rezultati = detector.detect(mp_image)

# Provera da li je detektovana bar jedna osoba/poza
if rezultati.pose_landmarks:
    # Uzimamo ključne tačke za prvu detektovanu osobu
    landmarks = rezultati.pose_landmarks[0]
    
    print("\n--- DETEKTOVANE KOORDINATE (33 TAČKE) ---")
    
    # Prolazak kroz svih 33 ključnih tačaka tela
    for i, lm in enumerate(landmarks):
        # Normalizovane koordinate (od 0.0 do 1.0) pretvaramo u stvarne piksele slike
        px = int(lm.x * sirina)
        py = int(lm.y * visina)
        
        # Crtamo pun krug (žutu tačku) na koordinatama zgloba: BGR format (0, 255, 255)
        cv2.circle(slika, (px, py), 6, (0, 255, 255), -1)
        
        # Ispisujemo najvažnije zglobove u terminal (nos, ramena, šake, kukovi, članci)
        if i in [0, 11, 12, 15, 16, 23, 24, 27, 28]:
            print(f"Tačka {i:>2}: X = {px}, Y = {py}")

    # ---------------------------------------------------------
    # 6. PRIKAZ REZULTATA NA EKRANU
    # ---------------------------------------------------------
    # Kreiranje prozora koji se može prilagođavati po veličini
    cv2.namedWindow("Korak 1 - Detekcija Tacaka", cv2.WINDOW_NORMAL)
    
    # Prikaz slike sa ucrtanim tačkama
    cv2.imshow("Korak 1 - Detekcija Tacaka", slika)
    
    # Čekanje da korisnik pritisne bilo koji taster pre zatvaranja prozora
    cv2.waitKey(0)
    
    # Zatvaranje svih otvorenih OpenCV prozora
    cv2.destroyAllWindows()

# ---------------------------------------------------------
# 7. OSLOBAĐANJE RESURSA
# ---------------------------------------------------------
# Zatvaranje detektora i oslobađanje memorije
detector.close()