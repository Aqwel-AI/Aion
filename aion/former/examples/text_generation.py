"""
Generate text with a transformer model (next-token sampling).

Uses a small untrained model by default; extend to load trained weights.
Run: python -m aion.former.examples.text_generation
"""
import os
import numpy as np

from aion.former.models import Transformer
from aion.former.datasets import TextDataset, SimpleTokenizer


def sample_logits(logits: np.ndarray, temperature: float = 1.0) -> int:
    """
    Sample next token from logits (last position) with temperature scaling.

    Parameters
    ----------
    logits : np.ndarray
        Logits for last position, shape (vocab_size,).
    temperature : float, optional
        Sampling temperature; lower = sharper (default 1.0).

    Returns
    -------
    int
        Sampled token id.
    """
    logits = logits[-1]
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    return int(np.random.choice(len(probs), p=probs))


def generate(
    model: Transformer,
    tokenizer: SimpleTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    seq_length: int = 64,
) -> str:
    """
    Generate text from a prompt using next-token sampling.

    Parameters
    ----------
    model : Transformer
        The transformer model (e.g. untrained or loaded).
    tokenizer : SimpleTokenizer
        Tokenizer with same vocab as model.
    prompt : str
        Initial text to continue.
    max_new_tokens : int, optional
        Number of tokens to generate (default 50).
    temperature : float, optional
        Sampling temperature (default 0.8).
    seq_length : int, optional
        Context length fed to the model (default 64).

    Returns
    -------
    str
        prompt + generated text (decoded).
    """
    ids = tokenizer.encode(prompt)
    if len(ids) > seq_length:
        ids = ids[-seq_length:]
    generated = list(ids)
    for _ in range(max_new_tokens):
        context = np.array([generated[-seq_length:]], dtype=np.int64)
        logits, _ = model.forward(context)
        next_id = sample_logits(logits._data[0], temperature=temperature)
        generated.append(next_id)
    return tokenizer.decode(generated)


def main():
    # Same sample text as train_small_model to match vocab
    SAMPLE = "To be or not to be that is the question " * 20
    seq_length = 64
    dataset = TextDataset(SAMPLE, seq_length=seq_length, level="char")
    model = Transformer(
        vocab_size=dataset.vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_len=seq_length,
    )
    prompt = "To be "
    print("Prompt:", repr(prompt))
    out = generate(
        model,
        dataset.tokenizer,
        prompt,
        max_new_tokens=60,
        temperature=0.9,
        seq_length=seq_length,
    )
    print("Generated:", out)


if __name__ == "__main__":
    main()
