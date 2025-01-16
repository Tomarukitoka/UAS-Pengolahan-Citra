import cv2
import numpy as np
import time
import sys
from collections import defaultdict

# Pastikan input ID pengguna diambil dari argumen
if len(sys.argv) < 2:
    print("Error: ID pengguna tidak diberikan!")
    sys.exit(1)

user_id = sys.argv[1]  # ID pengguna dari argumen
print(f"Debug: ID yang dimasukkan: {user_id}")

# Inisialisasi pengenalan wajah
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer.yml")  

name_list = ["", "Jean", "Hugo", "Lovina", "Elbert", "Weldon", "Zoe", "Caroline", "Rayhan", "Bryan", "Bagas", "Flo", "Jeremy", "Gilbert", "Jansen", "Kenneth", "Ruben", "Alfredo", "Michelle"]

def is_real_face(face_region):
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    is_real = laplacian_var > 50  
    if not is_real:
        return laplacian_var, False

    edges = cv2.Canny(gray_face, 100, 200)
    edge_count = np.sum(edges > 0)
    if edge_count < 100:
        return laplacian_var, False
    
    return laplacian_var, True

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Tidak dapat mengakses kamera!")
    sys.exit(1)

start_time = time.time()

face_counter = defaultdict(int) 
confidence_sum = defaultdict(int)  

detected_name = None
verification_success = False

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_region = frame[y:y+h, x:x+w]
        laplacian_var, is_real = is_real_face(face_region)

        if is_real:
            serial, conf_raw = recognizer.predict(gray[y:y+h, x:x+w])
            adjusted_conf = int(100 * (1 - conf_raw / 300))
            if serial >= len(name_list):
                name = "Unknown"
            else:
                name = name_list[serial]

            face_counter[name] += 1
            confidence_sum[name] += adjusted_conf

            if adjusted_conf > 75: 
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({adjusted_conf:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if name == user_id:  # Periksa apakah nama cocok dengan ID
                    detected_name = name
                    verification_success = True
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, f"Low Conf ({adjusted_conf:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, f"Spoofing! ({laplacian_var:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Face Recognition with Anti-Spoofing", frame)

    if time.time() - start_time > 10 or verification_success:
        print("Verifikasi selesai, menunggu hasil....")
        break  

    cv2.waitKey(1)

print("\nHasil Deteksi:")
if verification_success:
    print(f"Verifikasi berhasil! Nama: {detected_name}")
    print(f"ID cocok dengan input: {user_id}")
    print(detected_name)  # Kembalikan nama ke program lockscreen
else:
    print(f"Verifikasi gagal! Tidak ada kecocokan untuk ID: {user_id}")
    print("")

video.release()
cv2.destroyAllWindows()