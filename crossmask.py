from functools import partial

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def create_cross_attention_mask(
    input_ids1,
    attention_mask1,
    input_ids2,
    attention_mask2,
    tokenizer1,
    tokenizer2,
    block_size,
    visualization=False,
):
    """
    Create cross-attention masks between two tokenized sequences.
    When using tokenizer2 as the primary model, outputs masks in the order:
    [tokenizer2_primary_mask, tokenizer1_primary_mask]
    """
    batch_size = len(input_ids1)

    mask_2to1 = np.zeros((batch_size, block_size, block_size), dtype=bool)
    mask_1to2 = np.zeros((batch_size, block_size, block_size), dtype=bool)

    batch_tokens1 = []
    batch_tokens2 = []

    for b in range(batch_size):
        if visualization:
            tokens1 = [tokenizer1.decode([t]) for t in input_ids1[b]]
            batch_tokens1.append(tokens1)

            tokens2 = [tokenizer2.decode([t]) for t in input_ids2[b]]
            batch_tokens2.append(tokens2)

        valid_input_ids1 = input_ids1[b][attention_mask1[b] == 1]
        valid_input_ids2 = input_ids2[b][attention_mask2[b] == 1]

        valid_tokens1 = [tokenizer1.decode([t]) for t in valid_input_ids1]
        valid_tokens2 = [tokenizer2.decode([t]) for t in valid_input_ids2]

        # Calculate start positions for valid tokens (skipping special tokens and padding)
        start1 = attention_mask1[b].tolist().index(1)
        start2 = attention_mask2[b].tolist().index(1)

        # Create character-level mappings for tokens
        token1_to_char = create_token_to_char_mapping(
            valid_tokens1, tokenizer1, block_size, start1
        )
        token2_to_char = create_token_to_char_mapping(
            valid_tokens2, tokenizer2, block_size, start2
        )

        end1, end2 = token1_to_char[:, 1], token2_to_char[:, 1]
        mask_2to1[b] = (end1[np.newaxis, :] <= end2[:, np.newaxis]).astype(bool)
        mask_1to2[b] = (end2[np.newaxis, :] <= end1[:, np.newaxis]).astype(bool)

    if visualization:
        return {
            "mask_2to1": mask_2to1,
            "mask_1to2": mask_1to2,
            "batch_tokens1": batch_tokens1,
            "batch_tokens2": batch_tokens2,
            "context_attn_mask": np.concatenate(
                [mask_1to2, mask_2to1], axis=1, dtype=bool
            ),
        }
    else:
        return {
            "context_attn_mask": np.concatenate(
                [mask_1to2, mask_2to1], axis=1, dtype=bool
            )
        }


def create_token_to_char_mapping(tokens, tokenizer, block_size, start_idx):
    """
    Create a mapping from tokens to their character positions in the original text.

    Args:
        tokens: List of token strings
        tokenizer: The tokenizer used for tokenization
        block_size: Maximum sequence length
        start_idx: Starting index for valid tokens

    Returns:
        mapping: Array containing [start_pos, end_pos] for each token
    """
    mapping = np.full((block_size, 2), np.nan, dtype=np.float32)
    char_index = 0
    prev_token_text = ""
    is_first_unk_token = True

    for i, token_text in enumerate(tokens):
        # Handle consecutive unknown tokens
        if token_text == "�":
            if is_first_unk_token:
                mapping[start_idx + i] = np.array([char_index, char_index + 1])
                is_first_unk_token = False
            else:
                mapping[start_idx + i] = mapping[start_idx + i - 1]
        else:
            if prev_token_text == "�":
                char_index += 1
                is_first_unk_token = True

            if token_text == tokenizer.bos_token:
                mapping[start_idx + i] = np.array([0, 0])
            elif token_text == tokenizer.eos_token:
                mapping[start_idx + i] = np.array([char_index + 1, char_index + 1])
            else:
                # Handle special case of blank tokens at sequence start
                original_token_len = len(token_text)
                token_len = len(token_text.strip())
                # Ensure space tokens have minimum length of 1
                if original_token_len != 0 and token_len == 0:
                    token_len = 1
                mapping[start_idx + i] = np.array([char_index, char_index + token_len])
                char_index += token_len
        prev_token_text = token_text
    return mapping


def preprocess_strings(texts, tokenizer, block_size, add_special_tokens=True):
    """
    Preprocess text sequences by adding special tokens and tokenizing.

    Args:
        texts: List of input text strings
        tokenizer: Tokenizer to use for preprocessing
        block_size: Maximum sequence length
        add_special_tokens: Whether to add BOS/EOS tokens

    Returns:
        Dict containing processed texts and tokenization outputs
    """
    if add_special_tokens:
        bos_token = tokenizer.bos_token or tokenizer.cls_token or ""
        eos_token = tokenizer.eos_token or tokenizer.sep_token or ""

        processed_texts = [f"{bos_token}{text}{eos_token}" for text in texts]
    else:
        processed_texts = texts

    tokenized = tokenizer(
        processed_texts, padding="max_length", truncation=True, max_length=block_size
    )

    return {
        "processed_texts": processed_texts,
        "input_ids": np.asarray(tokenized.input_ids),
        "attention_mask": np.asarray(tokenized.attention_mask),
    }


def tokenize_function(examples, block_size, tokenizer1, tokenizer2):
    """
    Tokenize examples and create cross attention masks between the two tokenizations.
    """
    strings = examples.get("content", [])
    tokenizer1_preprocessed = preprocess_strings(
        strings, tokenizer=tokenizer1, block_size=block_size
    )
    tokenizer2_preprocessed = preprocess_strings(
        strings, tokenizer=tokenizer2, block_size=block_size
    )

    cross_attention_masks = create_cross_attention_mask(
        tokenizer1_preprocessed["input_ids"],
        tokenizer1_preprocessed["attention_mask"],
        tokenizer2_preprocessed["input_ids"],
        tokenizer2_preprocessed["attention_mask"],
        tokenizer1,
        tokenizer2,
        block_size,
    )
    return cross_attention_masks


def tokenize_datasets(files, block_size, tokenizer1, tokenizer2):
    """
    Tokenize and process multiple files from a dataset.
    """
    raw_datasets = load_dataset(
        "json",
        data_files=files,
    )
    tokenized_datasets = raw_datasets.map(
        partial(
            tokenize_function,
            block_size=block_size,
            tokenizer1=tokenizer1,
            tokenizer2=tokenizer2,
        ),
        batched=True,
        num_proc=64,
        batch_size=16,
        remove_columns="content",
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    return tokenized_datasets


if __name__ == "__main__":
    # Example usage
    example_files = [
        "./data/medical/textbooks.json",
    ]

    block_size = 4096

    # Initialize tokenizers
    tokenizer1 = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer2 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    tokenizer2.padding_side = "left"
    tokenizer2.pad_token = tokenizer2.eos_token

    result = tokenize_datasets(example_files, block_size, tokenizer1, tokenizer2)
