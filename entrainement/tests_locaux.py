"""Fait tourner tout le pipeline avec un mini-modèle et un générateur factices, sans GPU ni téléchargement.
Vérifie le code uniquement : les scores obtenus n'ont aucune valeur."""
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ICI = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp(prefix="detia-test-"))
env = {**os.environ, "DETIA_DATA": str(tmp / "donnees"), "DETIA_SORTIE": str(tmp / "sortie")}
rng = random.Random(0)

# 1. Textes « humains » factices, variés.
mots = ("le chat dort sur le canapé pendant que la pluie tombe dehors et je repense à ce voyage à Lyon "
        "mon frère disait toujours que les trains arrivent en retard mais ce jour là tout était à l'heure").split()
src = tmp / "humains_src.jsonl"
with src.open("w") as f:
    for i in range(240):
        phrases = [" ".join(rng.choice(mots) for _ in range(rng.randint(3, 25))).capitalize() + rng.choice([".", "!", "?", "..."]) for _ in range(rng.randint(8, 20))]
        f.write(json.dumps({"text": " ".join(phrases), "title": f"sujet {i}"}, ensure_ascii=False) + "\n")

# 2. Mini-modèle de base (BERT minuscule + tokenizer entraîné sur place).
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
tk.train([str(src)], trainers.WordPieceTrainer(vocab_size=800, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
from tokenizers.processors import TemplateProcessing
tk.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tk.token_to_id("[CLS]")), ("[SEP]", tk.token_to_id("[SEP]"))])
base = tmp / "base"
tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
tok.save_pretrained(base)
BertForSequenceClassification(BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                         intermediate_size=64, max_position_embeddings=520)).save_pretrained(base)


def run(*args):
    print("\n$", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ICI, env=env, check=True)


run("donnees.py", "--sources", "", "--local", str(src), "--domaine", "test")
run("generer.py", "--generateurs", "factice")
run("entrainer.py", "--base", str(base), "--epoques", "1", "--tours", "1", "--difficiles-par-tour", "20",
    "--generateurs", "factice", "--batch", "8", "--longueur", "128", "--sortie", str(tmp / "clf"))
run("exporter.py", "--source", str(tmp / "clf"), "--sortie", str(tmp / "clf-onnx"))
run("evaluer.py", "--classifieur", str(tmp / "clf"), "--classifieur-id", "test/test", "--heuristiques")

cal = json.loads((tmp / "sortie" / "js" / "calibration.json").read_text())
bench = json.loads((tmp / "sortie" / "resultats" / "benchmark.json").read_text())
assert "clf" in bench["methodes"] and "heur" in bench["methodes"], bench.keys()
assert "clf+heur" in cal["ensembles"], cal["ensembles"].keys()
assert (tmp / "clf-onnx" / "onnx" / "model_quantized.onnx").exists()
print("\nTOUS LES TESTS PASSENT. Dossier :", tmp)
print("Combinaisons réglées :", list(cal["ensembles"]))
