"""The SFT chat template must render exactly like Qwen's, and mask everything but the assistant replies."""

from pathlib import Path

import pytest
from transformers import AutoTokenizer

TRAINING_TEMPLATE = (Path(__file__).parents[1] / "chat_templates" / "qwen3_5_training.jinja").read_text()

CONVERSATIONS = {
    "single_turn": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ],
    "multi_turn_with_system": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ],
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")


@pytest.mark.parametrize("name", CONVERSATIONS)
def test_renders_like_original(tokenizer, name):
    messages = CONVERSATIONS[name]
    original = tokenizer.apply_chat_template(messages, tokenize=False)
    training = tokenizer.apply_chat_template(messages, tokenize=False, chat_template=TRAINING_TEMPLATE)
    assert training == original


@pytest.mark.parametrize("name", CONVERSATIONS)
def test_masks_assistant_replies_only(tokenizer, name):
    messages = CONVERSATIONS[name]
    encoded = tokenizer.apply_chat_template(
        messages, chat_template=TRAINING_TEMPLATE, return_dict=True, return_assistant_tokens_mask=True
    )
    trained = [token for token, mask in zip(encoded["input_ids"], encoded["assistant_masks"]) if mask]
    expected = "".join(m["content"] + "<|im_end|>" for m in messages if m["role"] == "assistant")
    assert tokenizer.decode(trained) == expected
