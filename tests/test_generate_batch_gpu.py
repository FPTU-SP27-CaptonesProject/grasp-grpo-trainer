"""Real-model check that left padding does not change greedy output. Opt-in: needs CUDA + an adapter."""
import os

import pytest

CHECKPOINT = os.environ.get("GRASP_TEST_CHECKPOINT")
pytestmark = pytest.mark.skipif(not CHECKPOINT, reason="set GRASP_TEST_CHECKPOINT to a local SFT adapter (needs CUDA)")


def test_batched_greedy_generation_matches_single_prompt():
    from grasp_grpo.model import generate, generate_batch, load_policy

    model, tokenizer, _ = load_policy(CHECKPOINT, trainable=False)
    prompts = ["Say yes.", "Name a colour.", "Count to three, separated by commas."]
    single = [generate(model, tokenizer, p, samples=1, max_new_tokens=16, temperature=0.0, top_p=1.0)[0].text for p in prompts]
    batched = [r[0].text for r in generate_batch(model, tokenizer, prompts, samples=1, max_new_tokens=16, temperature=0.0, top_p=1.0)]
    assert batched == single
