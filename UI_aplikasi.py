import os
import threading
import tkinter as tk
from tkinter import messagebox
from time import strftime
from PIL import Image, ImageTk
import time

def run_face_recognition(user_id):
    result = os.popen(f"python Final_TestMuka.py {user_id}").read().strip()
    print(f"Debug: Hasil dari Final_TestMuka.py: {result}")
    if result.startswith("Success:"):
        detected_name = result.split(":")[1].strip()
        return True, detected_name
    return False, None

def run_voice_recognition(user_id):
    print("Persiapan rekaman suara...")
    for i in range(3, 0, -1):  
        print(f"{i}...")
        time.sleep(1)

    print("Mulai merekam suara...")
    result = os.popen(f"python Final_TestSuara.py {user_id}").read().strip()
    print(f"Debug: Hasil dari Final_TestSuara.py: {result}")
    if result.startswith("Success"):
        return True
    return False

def verify():
    user_id = user_id_entry.get().strip()

    if not user_id:
        messagebox.showerror("Error", "Masukkan ID pengguna terlebih dahulu!")
        return

    result_face = None
    result_voice = None
    detected_name = None

    def face_thread():
        nonlocal result_face, detected_name
        try:
            result_face, detected_name = run_face_recognition(user_id)
            print(f"Debug: Hasil deteksi wajah: {result_face}, nama terdeteksi: {detected_name}")
        except Exception as e:
            print(f"Error dalam face_thread: {e}")
            result_face, detected_name = False, None

    def voice_thread():
        nonlocal result_voice
        try:
            result_voice = run_voice_recognition(user_id)
            print(f"Debug: Hasil deteksi suara: {result_voice}")
        except Exception as e:
            print(f"Error dalam voice_thread: {e}")
            result_voice = False

    t1 = threading.Thread(target=face_thread)
    t2 = threading.Thread(target=voice_thread)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    if result_face and detected_name == user_id and result_voice:
        unlock_screen(detected_name)
    else:
        messagebox.showerror("Verifikasi Gagal", "Wajah atau suara tidak sesuai, silahkan coba lagi")

def unlock_screen(detected_name):
    window = tk.Toplevel(root)
    window.title("Selamat Datang")
    window.geometry("400x600")
    window.configure(bg="#2c2c2c")

    welcome_label = tk.Label(window, text=f"Halo, {detected_name}", font=("Arial", 24), fg="white", bg="#2c2c2c")
    welcome_label.pack(pady=20)

    try:
        img_path = f"PP/PP{detected_name}.jpg"  
        img = Image.open(img_path)
        img = img.resize((300, 300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        img_label = tk.Label(window, image=img, bg="#2c2c2c")
        img_label.image = img  
        img_label.pack(pady=20)
    except FileNotFoundError:
        messagebox.showerror("Error", f"Gambar untuk {detected_name} tidak ditemukan!")

    window.mainloop()

def update_time():
    current_time = strftime("%H:%M")
    current_date = strftime("%a %d %b %Y")
    clock_label.config(text=current_time)
    date_label.config(text=current_date)
    root.after(1000, update_time)

root = tk.Tk()
root.title("Lock Screen")
root.geometry("400x600")
root.configure(bg="#2c2c2c")

clock_label = tk.Label(root, text="12:00", font=("Arial", 48), fg="white", bg="#2c2c2c")
clock_label.pack(pady=20)

date_label = tk.Label(root, text="Fri 16", font=("Arial", 18), fg="white", bg="#2c2c2c")
date_label.pack()

update_time()

user_id_label = tk.Label(root, text="Masukkan user ID", font=("Arial", 14), fg="white", bg="#2c2c2c")
user_id_label.pack(pady=10)

user_id_entry = tk.Entry(root, font=("Arial", 14))
user_id_entry.pack(pady=10)

unlock_button = tk.Button(root, text="Unlock", command=verify, font=("Arial", 18), bg="#4CAF50", fg="white", padx=20, pady=10)
unlock_button.pack(pady=20)

root.mainloop()