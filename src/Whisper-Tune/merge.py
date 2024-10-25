import os
import json
import random
import argparse

def load_and_merge_jsonl_files(file_paths):
    merged_data = []
    for file_path in file_paths:
        if os.path.isfile(file_path) and file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        if isinstance(item, dict) and 'path' in item and 'text' in item:
                            merged_data.append(item)
                        else:
                            print(f"Skipping invalid item in {file_path}: {item}")
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON in {file_path}: {line}")
    return merged_data

def shuffle_data(data):
    random.shuffle(data)
    return data

def save_to_jsonl(data, output_file):
    with open(output_file, 'w') as out_file:
        for item in data:
            json.dump(item, out_file)
            out_file.write('\n')

def main():
    parser = argparse.ArgumentParser(description='Merge and shuffle JSONL files for audio datasets.')
    parser.add_argument('files', metavar='F', type=str, nargs='+',
                        help='JSONL files to merge')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file for merged and shuffled JSONL')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling (default: 42)')

    args = parser.parse_args()

    # Set the random seed for reproducibility
    random.seed(args.seed)

    merged_data = load_and_merge_jsonl_files(args.files)
    shuffled_data = shuffle_data(merged_data)
    save_to_jsonl(shuffled_data, args.output)

    print(f"Merging and shuffling complete. Output saved to '{args.output}'.")
    print(f"Total number of items in the merged and shuffled dataset: {len(shuffled_data)}")
    print(f"Random seed used for shuffling: {args.seed}")

if __name__ == "__main__":
    main()