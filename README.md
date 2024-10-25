# Whisper Tune

<div align="center">

![Code in Progress](https://img.shields.io/badge/status-in_progress-red.svg) ![License](https://img.shields.io/github/license/LuluW8071/Whisper-Tune) ![Open Issues](https://img.shields.io/github/issues/LuluW8071/Whisper-Tune) ![Closed Issues](https://img.shields.io/github/issues-closed/LuluW8071/Whisper-Tune) ![Open PRs](https://img.shields.io/github/issues-pr/LuluW8071/Whisper-Tune) ![Repo Size](https://img.shields.io/github/repo-size/LuluW8071/Whisper-Tune) ![Last Commit](https://img.shields.io/github/last-commit/LuluW8071/Whisper-Tune)

</div>

This repository contains a script to fine-tune OpenAI's Whisper model for automatic speech recognition (ASR). The script allows for training on custom datasets using [__Mimic Recording Studio__](https://github.com/MycroftAI/mimic-recording-studio) and supports multiple Whisper model sizes (`tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `large-v3-turbo` or other pre-trained whisper models). It also allows for flexibility in batch size, gradient accumulation, learning rate, warmup steps, and number of epochs through command-line arguments.

<div align="center">

![Whisper](https://images.ctfassets.net/kftzwdyauwt9/d9c13138-366f-49d3-a1a563abddc1/8acfb590df46923b021026207ff1a438/asr-summary-of-model-architecture-desktop.svg)

</div>

## Features

- Fine-tune OpenAI Whisper models or custom pre-trained models that are finetuned from whisper models.
- Dataset handling with custom JSON files.
- Command-line interface (CLI) for adjusting model and training parameters.
- Support for __Comet ML__ and __TensorBoard__ for logging.
- Automatic model saving and pushing best model to Hugging Face Hub.

## Requirements

To install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

## Environment Variables

Ensure you have a `.env` file in the project root that contains your [__Comet ML__](https://www.comet.com/) API key for logging:

```
COMET_API_KEY = "your_comet_api_key"
```

The model training logs will be pushed to Comet ML for tracking the experiments.

## Usage

| Argument                        | Description                                                                                       | Default Value   |
|----------------------------------|---------------------------------------------------------------------------------------------------|-----------------|
| `--train_json`                   | Path to the training dataset in JSON format.                                                      | N/A             |
| `--test_json`                    | Path to the test dataset in JSON format.                                                          | N/A             |
| `--whisper_model`, `-model`                | Choose from `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`, `large-v3-turbo`, or provide a custom Whisper model name. | `base`             |
| `--batch_size`                   | The batch size for training and evaluation.                                                       | `16`             |
| `--gradient_accumulation_steps`, `-grad_steps`   | Number of gradient accumulation steps.                                                            | `1`             | 
| `--learning_rate`, `-lr`         | Learning rate for training.    | `2e-5`  |
| `--warmup_steps`                 | Number of warmup steps for the learning rate scheduler.                                            | `500`           |
| `--epochs`, `-e`                       | Number of epochs to train for.                                                                    | `10`            |
| `--num_workers`, `-w`            | Number of CPU workers.                                                                            | `2` |

This table concisely lists the arguments and their corresponding descriptions along with their default values.

```bash
python train.py \
    --train_json train_augmented.json \
    --test_json test.json \
    --whisper_model tiny \
    --batch_size 8 \
    --grad_steps 1 \
    --lr 2e-5 \
    --warmup_steps 200 \
    --epochs 5
    -w 4
```

## Pushing to Hugging Face Hub 🤗

The script is designed to __automatically push the best trained model to the Hugging Face Hub__. Make sure you have set up your Hugging Face credentials properly.

## Example Dataset Format

Your dataset JSON files should look like this:

```json
{"path": "path/to/audio/file1.wav", "text": "The transcription of the audio."}
{"path": "path/to/audio/file2.wav", "text": "Another transcription."}
```

> Ensure that the audio files are properly referenced.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

<!-- ## Contributions

Contributions, issues, and feature requests are welcome. Feel free to open a PR or an issue. -->