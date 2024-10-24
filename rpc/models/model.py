import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed.rpc as rpc
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    n_embd: int
    n_head: int
    n_layer: int
    batch_size: int
    block_size: int
    learning_rate: float
    dropout: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200

    @classmethod
    def get_config(cls, model_type: str):
        configs = {
            "shakespeare-200k": cls(
                vocab_size=65,
                batch_size=16,
                block_size=32,
                learning_rate=1e-3,
                n_embd=64,
                n_head=4,
                n_layer=4,
            ),
            "shakespeare-800k": cls(
                vocab_size=65,
                batch_size=32,
                block_size=64,
                learning_rate=6e-4,
                n_embd=128,
                n_head=4,
                n_layer=4,
            ),
            "shakespeare-10M": cls(
                vocab_size=65,
                batch_size=64,
                block_size=256,
                learning_rate=3e-4,
                n_embd=384,
                n_head=6,
                n_layer=6,
            ),
        }
        return configs.get(model_type)


class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.sa = MultiHeadAttention(config, config.n_head, head_size)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class FinalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            Block(config),
            Block(config),
            Block(config),
            Block(config),
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def to(self, device):
        super().to(device)
        self.device = device
        return self
