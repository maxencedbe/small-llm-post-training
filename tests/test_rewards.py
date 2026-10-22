"""GRPO rewards: correctness checks the final number, format only checks the '#### <number>' line."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from grpo import correctness_reward, format_reward  # noqa: E402


def completions(*texts):
    return [[{"role": "assistant", "content": text}] for text in texts]


def test_correctness_reward():
    texts = completions(
        "3 + 4 = 7\n#### 7",  # right
        "3 + 4 = 8\n#### 8",  # wrong number
        "The answer is 7.",  # no '####' line
        "#### 1,200",  # thousands separator
        "#### 7.",  # trailing dot
        "#### 6\nOops\n#### 7",  # the last answer counts
    )
    assert correctness_reward(texts, answer=["7", "7", "7", "1200", "7", "7"]) == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_format_reward_ignores_correctness():
    texts = completions("#### 8", "The answer is 7.", "#### -3.5")
    assert format_reward(texts) == [1.0, 0.0, 1.0]
