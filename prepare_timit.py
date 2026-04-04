import os
import subprocess
import zipfile
import shutil
import random
import re
from pathlib import Path


KAGGLE_DATASET = "mfekadu/darpa-timit-acousticphonetic-continuous-speech"
ZIP_FILE = "darpa-timit-acousticphonetic-continuous-speech.zip"
EXTRACT_DIR = "Data/Timit_extracted"

# Target directories defined in your pipeline
BASE_DIR = os.path.abspath("Data")
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
VAL_DIR = os.path.join(BASE_DIR, "Val")
TEST_DIR = os.path.join(BASE_DIR, "Test")
TEXT_DIR = os.path.join(BASE_DIR, "Text")
UNLABELLED_TEXT_FILE = os.path.join(TEXT_DIR, "unlabelled.txt")


def check_dependencies():
    """Check if kaggle and ffmpeg are installed."""
    if shutil.which("kaggle") is None:
        print("Error: 'kaggle' CLI is not installed. Run 'pip install kaggle' and setup your kaggle.json")
        exit(1)
    if shutil.which("ffmpeg") is None:
        print("Error: 'ffmpeg' is not installed. Run 'sudo apt-get install ffmpeg'")
        exit(1)


def download_and_extract():
    """Download the dataset from Kaggle if it doesn't exist and extract it."""
    if not os.path.exists(EXTRACT_DIR):
        if not os.path.exists(ZIP_FILE):
            print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
            subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET], check=True)
        else:
            print(f"Zip file {ZIP_FILE} already exists. Skipping download.")
        
        print(f"Extracting {ZIP_FILE}...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction complete.")
    else:
        print(f"Extracted directory '{EXTRACT_DIR}' already exists. Skipping download and extraction.")


def setup_directories():
    """Create the destination directories for the pipeline."""
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, TEXT_DIR]:
        os.makedirs(d, exist_ok=True)


def process_text(txt_path):
    """
    Extracts the pure text from a TIMIT transcript file.
    TIMIT .TXT format: '<start_sample> <end_sample> <text>'
    """
    with open(txt_path, 'r') as f:
        line = f.read().strip()
    
    # Split by spaces and grab everything after the two timestamp integers
    parts = line.split(" ", 2)
    if len(parts) == 3:
        raw_text = parts[2]
        # Remove punctuation, convert to upper case for consistency
        clean_text = re.sub(r'[^A-Za-z\s]', '', raw_text).upper().strip()
        return clean_text
    return ""


def convert_audio(input_wav, output_wav):
    """
    Converts NIST SPHERE or any WAV to 16kHz, Mono, standard RIFF WAV using ffmpeg.
    """
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(input_wav),
        "-acodec", "pcm_s16le", # Standard 16-bit PCM
        "-ar", "16000",         # 16 kHz sampling rate
        "-ac", "1",             # 1 channel (Mono)
        str(output_wav)
    ]
    subprocess.run(cmd, check=True)


def process_dataset():
    """Find, convert, and route the audio and text files."""
    print("Locating dataset files...")
    # TIMIT usually has a structure like data/TRAIN/ and data/TEST/
    all_wavs = list(Path(EXTRACT_DIR).rglob("*.[wW][aA][vV]"))
    
    if not all_wavs:
        print("No .WAV files found in the extracted directory!")
        exit(1)

    train_candidates = [w for w in all_wavs if "TRAIN" in str(w).upper()]
    test_candidates = [w for w in all_wavs if "TEST" in str(w).upper()]
    
    # Shuffle and split train to get a validation set (10% of train)
    random.seed(42)
    random.shuffle(train_candidates)
    val_split_idx = int(len(train_candidates) * 0.1)
    
    val_files = train_candidates[:val_split_idx]
    train_files = train_candidates[val_split_idx:]
    test_files = test_candidates
    
    splits = {
        "Train": (train_files, TRAIN_DIR),
        "Val": (val_files, VAL_DIR),
        "Test": (test_files, TEST_DIR)
    }

    all_text = []

    for split_name, (files, dest_dir) in splits.items():
        print(f"Processing {split_name} split ({len(files)} files)...")
        for wav_path in files:
            # Generate a unique filename using parent directory names to avoid collisions (e.g. DR1_FCJF0_SA1.wav)
            unique_name = "_".join(wav_path.parts[-3:]).lower()
            out_wav_path = os.path.join(dest_dir, unique_name)
            
            # Only convert if it doesn't already exist
            if not os.path.exists(out_wav_path):
                convert_audio(wav_path, out_wav_path)
            
            # Look for the corresponding .TXT file for transcriptions
            # TIMIT stores transcripts alongside wavs with the same base name
            txt_path = wav_path.with_suffix('.TXT')
            if not txt_path.exists():
                txt_path = wav_path.with_suffix('.txt')
            
            if txt_path.exists():
                sentence = process_text(txt_path)
                if sentence:
                    all_text.append(sentence)

    print(f"Writing {len(all_text)} sentences to {UNLABELLED_TEXT_FILE}...")
    with open(UNLABELLED_TEXT_FILE, 'w') as f:
        f.write("\n".join(all_text))
    
    print("\nData Preparation Complete! You can now run the pipeline:")
    print(f'./run_wav2vec.sh "{TRAIN_DIR}" "{VAL_DIR}" "{TEST_DIR}" "{UNLABELLED_TEXT_FILE}"')


if __name__ == "__main__":
    check_dependencies()
    setup_directories()
    download_and_extract()
    process_dataset()