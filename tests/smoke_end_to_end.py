"""
End-to-end smoke test on CPU: synthetic contracts + a tiny randomly initialised Llama with its own tokenizer.
Runs scripts 01 -> 02 -> 03 (incl. resume) -> 04. It checks that the pipeline RUNS and the files line up;
the tiny model's text quality is meaningless.   python tests/smoke_end_to_end.py
"""
import importlib.util, json, os, shutil, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
import yaml, torch
from test_data_pipeline import make_fake_dataset, hash_embed


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def make_tiny_llama(path, texts):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM
    specials = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=700, special_tokens=specials,
                           initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>")
    tok.chat_template = ("{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
                         "{{ m['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}"
                         "<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}")
    tok.save_pretrained(path)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      tie_word_embeddings=True, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    torch.manual_seed(0)
    LlamaForCausalLM(cfg).save_pretrained(path)


def main():
    tmp = tempfile.mkdtemp()
    try:
        data_root = os.path.join(tmp, "data"); make_fake_dataset(data_root, n_docs=24)
        corpus_texts = [open(os.path.join(data_root, "corpus", "cuad", f)).read() for f in os.listdir(os.path.join(data_root, "corpus", "cuad"))]
        model_dir = os.path.join(tmp, "tiny"); make_tiny_llama(model_dir, corpus_texts + ["Question Contract excerpts Answer"] * 20)
        cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "config.yaml")))
        cfg["paths"].update({"data_root_candidates": [data_root], "processed_dir": os.path.join(tmp, "processed"),
                             "output_dir": os.path.join(tmp, "outputs")})
        cfg["data"].update({"max_train_queries": 12, "max_val_queries": 4, "max_test_queries": 4})
        cfg["retrieval"].update({"chunk_size": 200, "chunk_overlap": 20})
        cfg["model"].update({"backbone": model_dir, "latent_dim": 8, "adapter_hidden": 16, "max_prompt_tokens": 400, "max_response_tokens": 30})
        cfg["generation_data"].update({"max_new_tokens": 12, "batch_size": 8})
        cfg["training"].update({"epochs": 2, "grad_accum": 2, "fp16": False, "log_every": 2, "kl_warmup_steps": 3})
        cfg["eval"].update({"num_samples": 3, "max_new_tokens": 12})
        cfg_path = os.path.join(tmp, "config.yaml"); yaml.safe_dump(cfg, open(cfg_path, "w"))
        os.chdir(tmp)

        print("\n### step 1: prepare data"); load_script("01_prepare_data").main(cfg_path, embed_fn=hash_embed)
        assert os.path.exists(os.path.join(cfg["paths"]["processed_dir"], "test.json"))
        print("\n### step 2: generate training data"); load_script("02_generate_training_data").main(cfg_path)
        rows = [json.loads(l) for l in open(os.path.join(cfg["paths"]["processed_dir"], "train_responses.jsonl"))]
        print("groups:", len(rows), "responses per group:", [len(r["responses"]) for r in rows][:5])
        assert rows and all(len(r["responses"]) >= 2 for r in rows)
        print("\n### step 3: train (2 epochs)"); t = load_script("03_train"); t.train(cfg)
        out = cfg["paths"]["output_dir"]; assert os.path.exists(os.path.join(out, "phase1_latent.pt"))
        log = json.load(open(os.path.join(out, "train_log.json"))); print("log entries:", len(log))
        print("\n### step 3b: resume (epochs=3 -> runs 1 more epoch)"); cfg["training"]["epochs"] = 3
        t.train(cfg, resume=True)
        assert json.load(open(os.path.join(out, "train_log.json")))[-1]["epoch"] == 3
        print("\n### step 4: evaluate"); load_script("04_evaluate").main(cfg_path)
        s = json.load(open(os.path.join(out, "eval_summary.json"))); assert set(s) == {"baseline", "latent"}
        smp = json.load(open(os.path.join(out, "eval_samples.json"))); assert len(smp[0]["latent"]) == 3
        print("\nSMOKE TEST PASSED")
    finally:
        os.chdir(ROOT); shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
