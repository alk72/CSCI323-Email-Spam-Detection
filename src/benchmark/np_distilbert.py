"""Minimal NumPy forward pass of a fine-tuned DistilBertForSequenceClassification
(used only where PyTorch is unavailable). Returns the spam logit difference."""
import numpy as np
from safetensors.numpy import load_file
from scipy.special import erf
from tokenizers import Tokenizer


def _ln(x, g, b, eps=1e-12):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


class NpDistilBert:
    def __init__(self, path, max_len=256):
        self.w = load_file(f"{path}/model.safetensors")
        self.tok = Tokenizer.from_file(f"{path}/tokenizer.json")
        # save_pretrained can store padding in tokenizer.json; logit() pads per batch itself
        # and takes len(ids) as the real length, so pre-padded ids would be attended to
        self.tok.no_padding()
        self.tok.enable_truncation(max_len)
        self.tok.no_padding()  # tokenizer.json pads to the batch; padding would count as real tokens
        self.max_len = max_len

    def _lin(self, x, name):
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def _forward(self, ids, mask):
        w, p = self.w, "distilbert."
        B, T = ids.shape
        x = w[p + "embeddings.word_embeddings.weight"][ids] + w[p + "embeddings.position_embeddings.weight"][:T]
        x = _ln(x, w[p + "embeddings.LayerNorm.weight"], w[p + "embeddings.LayerNorm.bias"])
        neg = np.where(mask[:, None, None, :] == 0, np.float32(-1e9), np.float32(0))
        H, dh = 12, 64
        for i in range(6):
            L = f"{p}transformer.layer.{i}."
            def heads(t):
                return t.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            q = heads(self._lin(x, L + "attention.q_lin")) / np.sqrt(dh)
            k = heads(self._lin(x, L + "attention.k_lin"))
            v = heads(self._lin(x, L + "attention.v_lin"))
            s = q @ k.transpose(0, 1, 3, 2) + neg
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            a = (s @ v).transpose(0, 2, 1, 3).reshape(B, T, H * dh)
            x = _ln(self._lin(a, L + "attention.out_lin") + x, w[L + "sa_layer_norm.weight"], w[L + "sa_layer_norm.bias"])
            h = self._lin(x, L + "ffn.lin1")
            h = 0.5 * h * (1 + erf(h / np.sqrt(2)))
            x = _ln(self._lin(h, L + "ffn.lin2") + x, w[L + "output_layer_norm.weight"], w[L + "output_layer_norm.bias"])
        c = np.maximum(self._lin(x[:, 0], "pre_classifier"), 0)
        logits = self._lin(c, "classifier")
        return logits[:, 1] - logits[:, 0]

    def logit(self, texts, batch=32):
        enc = self.tok.encode_batch(list(texts))
        order = np.argsort([len(e.ids) for e in enc])
        out = np.zeros(len(enc), dtype=np.float32)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            T = max(len(enc[i].ids) for i in idx)
            ids = np.zeros((len(idx), T), dtype=np.int64)
            mask = np.zeros((len(idx), T), dtype=np.int64)
            for r, i in enumerate(idx):
                n = len(enc[i].ids)
                ids[r, :n], mask[r, :n] = enc[i].ids, 1
            out[idx] = self._forward(ids, mask)
        return out
