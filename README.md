# Detekcija poze i koordinata

Python projekat koji koristi MediaPipe Pose Landmarker za prepoznavanje ključnih tačaka ljudskog tela na slici. OpenCV prikazuje sliku sa označenim tačkama i skeletom, a jedan primer računa i približnu udaljenost šake od nasumično postavljene mete.

## Sadržaj projekta

- `detekcijakoordinata.py` - detektuje 33 ključne tačke tela, ispisuje odabrane koordinate i prikazuje ih na slici.
- `spajanjekoordinata.py` - povezuje ključne tačke linijama i prikazuje skelet.
- `udaljenost.py` - prikazuje skelet, postavlja nasumičnu metu i procenjuje udaljenost desne šake od nje.
- `latinoposeslika.webp`, `poza1.jpg`, `poza2.jpg`, `poza3.jpg` - ulazne slike za primere.
- `keypoints_detection.task` i `pose_landmarker_heavy.task` - MediaPipe modeli.
- `instal.txt` - spisak osnovnih Python paketa.

## Zahtevi

- Python 3.10 ili noviji
- Internet konekcija pri prvom preuzimanju modela

Instalirajte potrebne pakete:

```bash
pip install opencv-python mediapipe numpy
```

## Pokretanje

Otvorite terminal u direktorijumu projekta i pokrenite željeni primer:

```bash
python detekcijakoordinata.py
```

Ostali primeri:

```bash
python spajanjekoordinata.py
python udaljenost.py
```

Programi otvaraju OpenCV prozor sa obrađenom slikom. Pritisnite bilo koji taster dok je prozor aktivan da biste ga zatvorili.

## Podešavanje slike

Promenljiva `PUTANJA_SLIKE` u Python skriptama trenutno sadrži apsolutnu putanju sa računara autora. Ako pokrećete projekat na drugom računaru, promenite je tako da pokazuje na željenu sliku, na primer:

```python
PUTANJA_SLIKE = "poza3.jpg"
```

Koristite naziv slike koja se nalazi u direktorijumu projekta. Imena modela u skriptama očekuju da se fajl `keypoints_detection.task` nalazi u direktorijumu iz kog pokrećete program. Skripta `detekcijakoordinata.py` ga preuzima automatski ako nedostaje.

## Napomena o proceni udaljenosti

`udaljenost.py` pretvara piksele u metre koristeći pretpostavljenu prosečnu visinu osobe od 1,75 m. Rezultat je gruba procena i nije kalibrisana mera stvarne udaljenosti.