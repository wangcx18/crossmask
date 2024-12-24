# CrossMask

A simple and efficient tool for generating cross-attention masks between different tokenized sequences. This is particularly useful for implementing cross-attention mechanisms in multi-model setups.

## Features

- Generate cross-attention masks between any two tokenizer outputs
- Support for batch processing
- Compatible with HuggingFace tokenizers

## Quick Start

Here's a basic example:

```python
from crossmask import create_cross_attention_mask
from transformers import AutoTokenizer

# Initialize tokenizers
tokenizer1 = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer2 = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Prepare input
text = "Hello world"
block_size = 512

# Tokenize text
tokens1 = tokenizer1(text, padding="max_length", truncation=True, max_length=block_size)
tokens2 = tokenizer2(text, padding="max_length", truncation=True, max_length=block_size)

# Generate cross attention mask
mask = create_cross_attention_mask(
    tokens1["input_ids"],
    tokens1["attention_mask"], 
    tokens2["input_ids"],
    tokens2["attention_mask"],
    tokenizer1,
    tokenizer2,
    block_size
)
```

## Contributing

Contributions are welcome! Feel free to submit Pull Requests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.