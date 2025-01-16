import cv2
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

id = input("Masukkan nomor ID: ")

base_dir = "datasetmuka"
sub_dir = os.path.join(base_dir, id)

os.makedirs(sub_dir, exist_ok=True)

existing_files = [f for f in os.listdir(sub_dir) if f.startswith("User.") and f.endswith(".jpg")]
existing_counts = [
    int(f.split(".")[2]) for f in existing_files if f.split(".")[2].isdigit()
]
start_count = max(existing_counts, default=0)  

count = start_count
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        count += 1
        file_path = os.path.join(sub_dir, f"User.{id}.{count}.jpg")
        cv2.imwrite(file_path, gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if count >= start_count + 200:  
        break

video.release()
cv2.destroyAllWindows()
print("Pengambilan dataset selesai..................")
