import os
from pydub import AudioSegment
from pydub.utils import make_chunks

def split_audio(input_folder, output_folder, chunk_duration_ms=4000):
    """
    Memotong file WAV menjadi beberapa bagian dengan durasi tertentu.

    Parameters:
        input_folder (str): Folder input yang berisi file WAV.
        output_folder (str): Folder output untuk menyimpan potongan audio.
        chunk_duration_ms (int): Durasi setiap potongan audio dalam milidetik (default 4000 ms / 4 detik).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".wav"):
                input_path = os.path.join(root, file)
                speaker_name = os.path.basename(root)

                # Folder khusus untuk setiap speaker
                speaker_output_folder = os.path.join(output_folder, speaker_name)
                if not os.path.exists(speaker_output_folder):
                    os.makedirs(speaker_output_folder)

                try:
                    # Load file WAV
                    audio = AudioSegment.from_wav(input_path)

                    # Memotong audio menjadi bagian-bagian kecil
                    chunks = make_chunks(audio, chunk_duration_ms)

                    # Simpan setiap potongan ke dalam folder output
                    for i, chunk in enumerate(chunks):
                        output_file = os.path.join(speaker_output_folder, f"{os.path.splitext(file)[0]}_part{i+1}.wav")
                        chunk.export(output_file, format="wav")

                    print(f"File {file} berhasil dipotong dan disimpan di {speaker_output_folder}")
                except Exception as e:
                    print(f"Gagal memproses {file}: {e}")

if __name__ == "__main__":
    # Folder input berisi file audio
    input_folder = "tes"

    # Folder output untuk menyimpan potongan audio
    output_folder = "output_audio"

    # Durasi setiap potongan audio dalam milidetik (4 detik)
    chunk_duration_ms = 4000

    print("Memotong file audio...")
    split_audio(input_folder, output_folder, chunk_duration_ms)
    print("Pemotongan audio selesai.")
