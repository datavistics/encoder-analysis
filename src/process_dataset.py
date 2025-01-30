import base64
import json
import os
import random
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from loguru import logger


def tokenize_and_filter(dataset, tokenizer, text_column, min_tokens=None, max_tokens=None, num_proc=8):
    """Tokenizes dataset, adds a num_tokens column, and filters based on token length constraints."""

    # Tokenize and count tokens
    dataset = dataset.map(
            lambda example: {"num_tokens": len(tokenizer.encode(example[text_column]))},
            num_proc=num_proc,
            )

    # Apply filtering if min_tokens and max_tokens are provided
    if min_tokens is not None and max_tokens is not None:
        dataset = dataset.filter(lambda x: min_tokens <= x["num_tokens"] <= max_tokens, num_proc=num_proc)

    return dataset


def sample_dataset(dataset, n_samples, seed=42):
    """Samples a dataset randomly if it has more than n_samples."""
    total_samples = len(dataset)

    if total_samples <= n_samples:
        return dataset

    random.seed(seed)
    random_indices = random.sample(range(total_samples), n_samples)
    return_dataset = dataset.select(random_indices)
    logger.success(f"Sampled dataset down to {len(return_dataset)} samples")

    return return_dataset


def save_dataset(data, file_path):
    """Saves processed dataset in JSON or JSONL format based on file extension."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Convert to a list of dictionaries
    data = data.to_list()

    if file_path.endswith(".jsonl"):
        # Save as JSONL (one JSON object per line)
        with open(file_path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Saved dataset to {file_path} in JSONL format")
    elif file_path.endswith(".json"):
        # Save as a JSON array
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Saved dataset to {file_path} in JSON format")
    else:
        raise ValueError("Unsupported file extension. Use '.json' or '.jsonl'.")


def load_json_files(folder_path):
    all_data = []
    folder = Path(folder_path)

    # Iterate over all JSON files in the folder
    for file_path in folder.glob("*/*.json"):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

                # If data is a list of dicts, extend it
                if isinstance(data, list):
                    all_data.extend(data)
                # If data is a single dict, append it as a row
                elif isinstance(data, dict):
                    all_data.append(data)
                else:
                    logger.warning(f"Skipping {file_path.name}: Unexpected format")
        except json.JSONDecodeError:
            logger.error(f"Skipping {file_path.name}: Invalid JSON")

    # Convert to DataFrame
    return pd.DataFrame(all_data)


def pil_to_base64(image: Image.Image, format: str = "PNG", modality: str = "image") -> str:
    """
    Converts a PIL image to a base64-encoded data URI.

    :param image: PIL Image object
    :param format: Image format (e.g., "PNG", "JPEG")
    :param modality: MIME type category (default: "image")
    :return: Base64-encoded data URI
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    base64_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mimetype = f"{modality}/{format.lower()}"
    return f"data:{mimetype};base64,{base64_encoded}"
