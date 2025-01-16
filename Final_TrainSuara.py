import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import pickle

def load_audio(audio_path, sr=22050):
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        return y
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

def extract_mel_spectrogram(y, sr=22050):
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram

def extract_features(audio_path):
    try:
        y = load_audio(audio_path)
        if y is not None:
            mel_spec = extract_mel_spectrogram(y)
            mel_spec_mean = np.mean(mel_spec, axis=1)
            return mel_spec_mean
        else:
            return None
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def load_dataset(data_path):
    features = []
    labels = []

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                mel_spec_mean = extract_features(file_path)
                
                if mel_spec_mean is not None:
                    features.append(mel_spec_mean)
                    labels.append(root.split(os.sep)[-1])  
                
    return np.array(features), np.array(labels)

def train_model(features, labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=50)  
    features_pca = pca.fit_transform(features_scaled)


    X_train, X_test, y_train, y_test = train_test_split(features_pca, labels_encoded, test_size=0.2, random_state=42)

    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi Model: {acc * 100:.2f}%")

    return model, le, scaler, pca

def save_model(model, label_encoder, scaler, pca, model_path="voice_recognition_model.pkl"):
    with open(model_path, "wb") as file:
        pickle.dump({"model": model, "label_encoder": label_encoder, "scaler": scaler, "pca": pca}, file)

if __name__ == "__main__":
    dataset_path = "datasetsuara" 
    print("Loading dataset...")
    
    features, labels = load_dataset(dataset_path)

    if len(features) == 0:
        print("No valid audio data found. Exiting...")
        exit()

    print(f"Total {len(features)} samples.")
    print("Training model...")
    model, label_encoder, scaler, pca = train_model(features, labels)
    save_model(model, label_encoder, scaler, pca)
    print("Model Tersimpan.")
