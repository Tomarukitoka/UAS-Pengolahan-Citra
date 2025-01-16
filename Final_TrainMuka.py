import cv2
import os
import numpy as np

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()

dataset_path = "datasetmuka" 
faces = []
labels = []

if not os.path.exists(dataset_path):
    print(f"Folder {dataset_path} tidak ditemukan. Pastikan folder dataset ada.")
    exit()

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    
    if os.path.isdir(label_path) and label.isdigit():
        image_count = 0 
        for image_name in os.listdir(label_path):
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                if image_count >= 150: 
                    break
                
                image_path = os.path.join(label_path, image_name)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Gambar {image_path} gagal dimuat.")
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                

                detected_faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
                
                if len(detected_faces) == 0:
                    print(f"Tidak ada wajah yang terdeteksi dalam gambar {image_path}.")
                    continue

                for (x, y, w, h) in detected_faces:
                    face_region = gray[y:y+h, x:x+w]
                    faces.append(face_region)
                    labels.append(int(label))  # Pastikan label adalah angka yang sesuai
                image_count += 1
                print(f"Menambahkan gambar dari {image_path} untuk label {label}")
    
# Pastikan data pelatihan ada
if len(faces) == 0:
    print("Tidak ada gambar wajah yang ditemukan untuk pelatihan.")
    exit()

# Latih model dengan data wajah dan label
recognizer.train(faces, np.array(labels))

# Simpan model yang sudah dilatih
recognizer.save("Trainer.yml")
print("Model telah disimpan sebagai Trainer.yml")
