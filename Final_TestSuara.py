import sys
import sounddevice as sd
import librosa
import numpy as np
import pickle

def record_audio(duration=3, sr=22050):
    print("Memulai rekaman... silahkan berbicara")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait() 
    print("Rekaman selesai")
    return audio.flatten(), sr

def load_saved_model(model_path="voice_recognition_model.pkl"):
    with open(model_path, "rb") as file:
        data = pickle.load(file)
    return data["model"], data["label_encoder"], data["scaler"], data["pca"]

def extract_features(y, sr, n_mels=128):
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_mean = np.mean(mel_spec, axis=1)
    return mel_spec_mean

def predict_audio_from_mic(model, label_encoder, scaler, pca, duration=3, n_mels=128):
    try:
        y, sr = record_audio(duration=duration)
        features = extract_features(y, sr, n_mels=n_mels)
        if features is None:
            return None, None

        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)
        prediction = model.predict(features_pca)
        probabilities = model.predict_proba(features_pca)

        predicted_label = label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities) * 100
        return predicted_label, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None

def verify_user_voice(user_id):
    model, label_encoder, scaler, pca = load_saved_model("voice_recognition_model.pkl")
    predicted_label, confidence = predict_audio_from_mic(model, label_encoder, scaler, pca)
    
    if predicted_label:
        print(f"Prediksi Suara: {predicted_label}")
        print(f"Confidence: {confidence:.2f}%")
    
        id_to_label = {
            "1": "Jean", "2": "Hugo", "3": "Lovina", "4": "Elbert",
            "5": "Weldon", "6": "Zoe", "7": "Caroline", "8": "Rayhan",
            "9": "Bryan", "10": "Bagas", "11": "Flo", "12": "Jeremy",
            "13": "Gilbert", "14": "Jansen", "15": "Kenneth", "16": "Ruben", "17": "Alfredo", "18":"Michelle"
        }

        if id_to_label.get(user_id) == predicted_label:
            print(f"Verifikasi untuk {predicted_label} berhasil")
            return True
        else:
            print("Verifikasi gagal.")
            return False
    else:
        print("Tidak ada prediksi.")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ID user tidak ada!")
        sys.exit(1)

    user_id = sys.argv[1]
    if verify_user_voice(user_id):
        print("Suara cocok!")
    else:
        print("Suara tidak cocok!")
