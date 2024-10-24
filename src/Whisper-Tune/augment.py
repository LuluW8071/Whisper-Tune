import os
import argparse
import json
import random
import numpy as np
import subprocess
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import soundfile as sf
import librosa

# Noise function using numpy and librosa
def noise(audio_data, sr, noise_rate=0.028):
    noise_amp = noise_rate * np.random.uniform() * np.amax(audio_data)
    audio_data = audio_data + noise_amp * np.random.normal(size=audio_data.shape[0])
    return audio_data, sr

# Stretch function using librosa
def stretch(audio_data, sr, rate=0.8):
    return librosa.effects.time_stretch(y=audio_data, rate=rate), sr

# Shift function using numpy
def shift(audio_data, sr):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1700)
    return np.roll(audio_data, shift_range), sr

# Pitch function using librosa
def pitch(audio_data, sr, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=audio_data, sr=sr, n_steps=pitch_factor), sr

# Process file function for multi-threading
def process_file(row, base_directory, output_directory):
    file_name = row['path']
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    audio_path = os.path.join(base_directory, file_name)

    # Load audio using librosa
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    # Prepare the list of processed data with the original file
    processed_data = [{
        'path': os.path.abspath(audio_path),  # No conversion needed
        'text': row['text']
    }]

    # Apply noise, stretch, shift, pitch
    augmentations = {
        'noise': noise,
        'stretch': stretch,
        'shift': shift,
        'pitch': pitch
    }

    for aug_name, aug_func in augmentations.items():
        try:
            augmented_data, sr = aug_func(audio_data, sr)
            output_name = f"{base_name}_{aug_name}.flac"
            output_path = os.path.join(output_directory, output_name)
            sf.write(output_path, augmented_data, sr, format='flac')  # Use soundfile to write FLAC files
            processed_data.append({
                'path': os.path.abspath(output_path),
                'text': row['text']
            })
        except Exception as e:
            print(f"Error applying {aug_name} augmentation to {audio_path}: {e}")

    return processed_data


def main(args):
    base_directory = os.path.dirname(args.input_file)
    clips_directory = os.path.abspath(os.path.join(args.save_json_path, 'clips'))

    if not os.path.exists(clips_directory):
        os.makedirs(clips_directory)

    # Read the JSON file line by line
    data = []
    with open(args.input_file, 'r', encoding="utf-8") as file:
        for line in file:
            if line.strip():  # Check if the line is not empty
                try:
                    data.append(json.loads(line.strip()))  # Load each line as a JSON object
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {line}. Error: {e}")

    print(f"{len(data)} files found. Processing using {args.num_workers} workers.")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(executor.map(lambda x: process_file(x, base_directory, clips_directory), data), total=len(data)))

    # Flatten the list of results
    all_data = [item for sublist in results for item in sublist if sublist]
    random.shuffle(all_data)

    with open(os.path.join(args.save_json_path, 'train_augmented.json'), 'w', encoding='utf-8') as json_file:
        for item in all_data:
            json_file.write(json.dumps(item) + '\n')

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
            Utility script to process audio files using noise, stretch, shift, pitch augmentations and save as JSON with file path and text."""
    )
    parser.add_argument('--input_file', type=str, default=None, required=True, help='Path to the input JSON file containing audio file paths and text')
    parser.add_argument('--save_json_path', type=str, default='', required=False, help='Path to the directory where the augmented JSON file will be saved')
    parser.add_argument('-w', '--num_workers', type=int, default=2, help='Number of worker threads for processing')

    args = parser.parse_args()
    main(args)
