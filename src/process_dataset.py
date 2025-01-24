import json
import os
import random

from loguru import logger


def tokenize_and_filter(dataset, tokenizer, text_column, num_proc=8):
    """Tokenizes dataset and adds a num_tokens column."""
    return dataset.map(
            lambda example: {"num_tokens": len(tokenizer.encode(example[text_column]))},
            num_proc=num_proc,
            )


def sample_dataset(dataset, n_samples, min_tokens, max_tokens, seed=42):
    """Filters and samples a dataset based on token length constraints."""
    filtered_dataset = dataset.filter(
            lambda x: min_tokens <= x["num_tokens"] <= max_tokens, num_proc=8
            )

    total_samples = len(filtered_dataset)
    if total_samples <= n_samples:
        return filtered_dataset

    random.seed(seed)
    random_indices = random.sample(range(total_samples), n_samples)
    return_dataset = filtered_dataset.select(random_indices)
    logger.success(f"Sampled dataset down to {len(return_dataset)} samples")
    return return_dataset


def save_dataset(data, file_path):
    """Saves processed dataset to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Convert to a list of dictionaries
    data = data.to_list()

    # Save as a JSON array (pretty formatted for readability)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.success(f"Saved dataset to {file_path}")
