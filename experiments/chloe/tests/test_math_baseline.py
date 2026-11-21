from copy import deepcopy
import json
import random
import os

import pytest
import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

import common
from common import attach_baseline, collate, digest, encode_example, parameter_report, seed_everything, validate_config
from evaluate import generate_completion, numeric_answer, score_records
from prepare_data import prepare
from train import fit, require_gpu_allocation


@pytest.fixture
def config():
    return json.loads((common.ROOT / "experiments/chloe/configs/smoke-lora-r8.json").read_text())


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text, *, add_special_tokens):
        return ([1] if add_special_tokens else []) + [3 + ord(c) % 50 for c in text]

    def decode(self, ids, *, skip_special_tokens):
        return " ".join(str(int(i)) for i in ids)


def test_response_mask_keeps_eos_and_ignores_only_padding():
    tokenizer = Tokenizer()
    short = encode_example({"query": "1", "response": "2"}, tokenizer, template="Q:{query}\nA:", max_length=20, max_prompt_length=10)
    long = encode_example({"query": "9" * 30, "response": "3" * 30}, tokenizer, template="Q:{query}\nA:", max_length=20, max_prompt_length=10)
    batch = collate([short, long], tokenizer.pad_token_id)
    assert short["labels"][-1] == tokenizer.eos_token_id
    assert batch["labels"][0, len(short["labels"]) - 1] == tokenizer.eos_token_id
    assert batch["labels"][0, len(short["labels"]):].eq(-100).all()
    assert batch["attention_mask"][0, len(short["labels"]):].eq(0).all()
    assert long["input_ids"][0] == 1
    assert long["prompt_truncated"] and long["response_truncated"]
    assert len(long["input_ids"]) == 20
    assert sum(v != -100 for v in long["labels"]) == 10


def test_manifest_is_deterministic_and_never_overwrites_different_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "OUTPUT_ROOT", tmp_path)
    source = tmp_path / "source.json"
    source.write_text(json.dumps([{"query": str(i), "response": str(i + 1)} for i in range(20)]))
    output = tmp_path / "manifest.json"
    first = prepare(source, count=8, seed=42, expected_rows=20, revision="fixture", output=output)
    assert first == prepare(source, count=8, seed=42, expected_rows=20, revision="fixture", output=output)
    assert len(set(first["indices"])) == 8
    with pytest.raises(ValueError, match="different source/seed"):
        prepare(source, count=8, seed=43, expected_rows=20, revision="fixture", output=output)
    changed = deepcopy(first)
    changed["indices"][0] = changed["indices"][1]
    output.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="corrupted"):
        prepare(source, count=8, seed=42, expected_rows=20, revision="fixture", output=output)


@pytest.mark.parametrize("text,expected", [("Answer: 1,200", "1200"), ("#### -3.50", "-7/2"),
                                          (r"\boxed{\frac{1}{2}}", "1/2"), ("result .5", "1/2"),
                                          ("1e3", "1000"), ("no answer", None), ("#### 1/0", None)])
def test_numeric_extraction(text, expected):
    answer = numeric_answer(text)
    assert (str(answer) if answer is not None else None) == expected


def test_accuracy_uses_all_cases_including_unparseable_predictions():
    scores, rows = score_records([{"source_index": 0, "gold": "work #### 1,200", "prediction": "1200"},
                                  {"source_index": 1, "gold": "work #### 2", "prediction": "no answer"}])
    assert scores["accuracy"] == 0.5 and scores["evaluation_examples"] == 2
    assert rows[1]["correct"] is False
    with pytest.raises(ValueError, match="Duplicate"):
        score_records([rows[0], rows[0]])


def test_code_protocol_and_formal_math_are_blocked_before_model_loading(config):
    with pytest.raises(ValueError, match="Code protocol is pending"):
        validate_config(common.read_json("experiments/chloe/configs/code-route.pending.json"))
    config["run_type"] = "formal"
    with pytest.raises(ValueError, match="Formal Math protocol"):
        validate_config(config)


def test_real_gpu_entrypoint_rejects_login_node(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="approved SLURM allocation"):
        require_gpu_allocation()


def test_shared_pretrained_entry_loads_local_tiny_fixture_offline(config, tmp_path):
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2))
    model.save_pretrained(tmp_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=BackendTokenizer(WordLevel(
        {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token="[UNK]")),
        unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]")
    tokenizer.save_pretrained(tmp_path)
    config["model"].update(path=str(tmp_path), revision="local-fixture")
    loaded, loaded_tokenizer = common.load_local_model(config, "cpu")
    assert next(loaded.parameters()).dtype == torch.bfloat16
    assert loaded_tokenizer.pad_token_id == loaded_tokenizer.eos_token_id == 2
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def tiny_run_setup(config, *, attention_dropout=0.0):
    config = deepcopy(config)
    config["adapter"]["rank"] = 2
    config["training"].update(epochs=1, max_steps=4, micro_batch_size=1,
                               gradient_accumulation=2, global_batch_size=2, warmup_steps=0,
                               learning_rate=0.003, checkpoint_every=2, log_every=100)
    seed_everything(73)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      max_position_embeddings=64, attention_dropout=attention_dropout))
    examples = []
    for i in range(8):
        prompt = [1, 3 + i]
        response = [12 + i] * (i % 4 + 1) + [2]
        examples.append({"input_ids": prompt + response, "labels": [-100] * len(prompt) + response})

    def make():
        seed_everything(97)
        model = deepcopy(base)
        plan = attach_baseline(model, config)
        if config["training"]["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return model, plan

    return config, examples, make


def test_generation_uses_prompt_tail_and_returns_only_completion(config):
    _, _, make = tiny_run_setup(config)
    model, _ = make()
    model.eval()
    result = generate_completion(model, Tokenizer(), "question " * 12, max_prompt_length=8,
                                 generation_config=GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=2,
                                                                    eos_token_id=2, pad_token_id=2, use_cache=True))
    assert result["prompt_truncated"] and len(result["prompt_input_ids"]) == 8
    assert 1 <= len(result["prediction"].split()) <= 2


def test_accumulated_gradient_matches_full_batch_with_unequal_response_lengths(config, tmp_path):
    config, examples, make = tiny_run_setup(config)
    small, plan = make()
    fit(small, examples, plan, config, pad_token_id=2, checkpoint=tmp_path / "accum.pt", fingerprint="accum")
    large, plan = make()
    full = deepcopy(config)
    full["training"].update(micro_batch_size=2, gradient_accumulation=1)
    fit(large, examples, plan, full, pad_token_id=2, checkpoint=tmp_path / "batch.pt", fingerprint="batch")
    for left, right in zip(small.parameters(), large.parameters()):
        torch.testing.assert_close(left, right, atol=3e-7, rtol=3e-5)


def test_restart_restores_cursor_optimizer_rng_and_freeze(config, tmp_path):
    config, examples, make = tiny_run_setup(config, attention_dropout=0.1)
    full, plan = make()
    frozen = {n: p.detach().clone() for n, p in full.named_parameters() if not p.requires_grad}
    report = parameter_report(full, plan)
    assert report["total_trainable_parameters"] == report["adapter_parameters"]
    assert report["unique_trainable_tensors"] == 2 * 2 * 4
    fit(full, examples, plan, config, pad_token_id=2, checkpoint=tmp_path / "full.pt", fingerprint="same")
    next_python, next_torch = random.random(), torch.rand(4)
    part, plan = make()
    path = tmp_path / "restart.pt"
    outcome = fit(part, examples, plan, config, pad_token_id=2, checkpoint=path, fingerprint="same", stop_after_step=2)
    assert outcome["status"] == "checkpointed"
    state = torch.load(path, weights_only=False)
    assert all(".down." in name or ".up." in name for name in state["adapter_state"])
    continued, plan = make()
    resumed = fit(continued, examples, plan, config, pad_token_id=2, checkpoint=path, fingerprint="same", resume=path)
    assert resumed["status"] == "completed" and resumed["progress"]["global_step"] == 4
    assert random.random() == next_python
    torch.testing.assert_close(torch.rand(4), next_torch, atol=0, rtol=0)
    for (name, actual), expected in zip(continued.named_parameters(), full.parameters()):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        if name in frozen:
            torch.testing.assert_close(actual, frozen[name], atol=0, rtol=0)
    again, plan = make()
    with pytest.raises(ValueError, match="fingerprint"):
        fit(again, examples, plan, config, pad_token_id=2, checkpoint=path, fingerprint="changed", resume=path)
