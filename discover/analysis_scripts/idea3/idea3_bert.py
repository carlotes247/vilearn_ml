"""Idea3 step 2b: German BERT frozen embeddings per group-window (generic-text baseline,
contrast to the rubric-guided LLM). deepset/gbert-base, frozen, mean-pooled (attention-mask
weighted) last hidden state -> 768-d. LOCAL ONLY.

Writes idea3_bert_embeddings.npy (n x 768) + idea3_bert_index.csv (aligned row-for-row
with idea3_window_text.csv). Empty/silence -> encoding of "(silence — no speech)".
"""
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "deepset/gbert-base"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("scratch/ideas/idea3_window_text.csv")
texts = [t if isinstance(t, str) and t.strip() else "(silence — no speech)" for t in df["text"]]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(DEV).eval()

embs = []
with torch.no_grad():
    for i in range(0, len(texts), 16):
        batch = texts[i:i + 16]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=512, padding=True).to(DEV)
        out = model(**enc).last_hidden_state            # (B, T, 768)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # mean-pool over real tokens
        embs.append(pooled.cpu().numpy())
        print(f"{min(i + 16, len(texts))}/{len(texts)}", flush=True)

E = np.vstack(embs)
np.save("scratch/ideas/idea3_bert_embeddings.npy", E)
df[["group_name", "sec", "y", "our_te"]].to_csv("scratch/ideas/idea3_bert_index.csv", index=False)
print(f"wrote idea3_bert_embeddings.npy {E.shape}; NaNs={np.isnan(E).sum()}")
