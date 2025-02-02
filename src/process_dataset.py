import base64
import json
import os
import random
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
from datasets import Dataset
from loguru import logger


def tokenize_and_filter(dataset: Dataset, tokenizer, text_column: str, min_tokens: int = None, max_tokens: int = None,
                        num_proc: int = 8):
    """
    Tokenizes a dataset, adds a `num_tokens` column, and filters based on token length constraints.

    :param dataset: Dataset to tokenize and filter.
    :param tokenizer: Tokenizer object with an `encode` method.
    :param text_column: Column name containing text data.
    :param min_tokens: Minimum number of tokens for filtering (optional).
    :param max_tokens: Maximum number of tokens for filtering (optional).
    :param num_proc: Number of processes for parallel execution.
    :return: Filtered dataset with token counts.
    """
    logger.info("Tokenizing dataset and applying token count filter")

    dataset = dataset.map(
            lambda example: {"num_tokens": len(tokenizer.encode(example[text_column]))},
            num_proc=num_proc,
            )

    if min_tokens is not None and max_tokens is not None:
        dataset = dataset.filter(lambda x: min_tokens <= x["num_tokens"] <= max_tokens, num_proc=num_proc)
        logger.info(f"Filtered dataset with token range [{min_tokens}, {max_tokens}]")

    return dataset


def sample_dataset(dataset, n_samples: int, seed: int = 42):
    """
    Samples a dataset randomly if it has more than `n_samples`.

    :param dataset: Dataset to sample from.
    :param n_samples: Number of samples to retain.
    :param seed: Random seed for reproducibility.
    :return: Sampled dataset.
    """
    total_samples = len(dataset)

    if total_samples <= n_samples:
        return dataset

    random.seed(seed)
    random_indices = random.sample(range(total_samples), n_samples)
    return_dataset = dataset.select(random_indices)
    logger.success(f"Sampled dataset down to {len(return_dataset)} samples")

    return return_dataset


def save_dataset(data, file_path: str):
    """
    Saves a dataset in JSON or JSONL format based on the file extension.

    :param data: Dataset to save.
    :param file_path: Path where the dataset should be saved.
    """
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
        logger.error("Unsupported file extension. Use '.json' or '.jsonl'.")
        raise ValueError("Unsupported file extension. Use '.json' or '.jsonl'.")


def load_json_files(folder_path: str) -> pd.DataFrame:
    """
    Loads JSON files from a folder into a Pandas DataFrame.

    :param folder_path: Path to the folder containing JSON files.
    :return: DataFrame containing all loaded data.
    """
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

    logger.info(f"Loaded {len(all_data)} entries from {folder_path}")
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
