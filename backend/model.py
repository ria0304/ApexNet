import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.GELU(),
            nn.Conv1d(48, hidden, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, t, f = x.shape
        h = x.reshape(b * n, t, f).transpose(1, 2)
        h = self.net(h).squeeze(-1)
        return h.view(b, n, -1)


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, 1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, 1, heads, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.W(x).view(b, n, self.heads, self.out_dim)
        e_src = (h * self.att_src).sum(-1)
        e_dst = (h * self.att_dst).sum(-1)
        scores = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))
        mask = adj.unsqueeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        alpha = torch.softmax(scores, dim=2)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)
        out = torch.einsum("bijn,bjnd->bind", alpha, h)
        return out.reshape(b, n, self.heads * self.out_dim)


class PairHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 1),
        )

    def forward(self, h: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        d = h.size(-1)
        attacker = torch.gather(h, 1, pairs[:, :, 0].unsqueeze(-1).expand(-1, -1, d))
        defender = torch.gather(h, 1, pairs[:, :, 1].unsqueeze(-1).expand(-1, -1, d))
        feat = torch.cat([attacker, defender, attacker - defender], dim=-1)
        return self.mlp(feat).squeeze(-1)


class CNNGAT(nn.Module):
    def __init__(self, in_features: int, hidden: int = 48, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.cnn = TemporalCNN(in_features, hidden)
        self.gat1 = GATLayer(hidden, hidden // heads, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.gat2 = GATLayer(hidden, hidden // heads, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.head = PairHead(hidden)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x)
        h = self.norm1(h + F.elu(self.gat1(h, adj)))
        h = self.norm2(h + F.elu(self.gat2(h, adj)))
        return h

    def forward(self, x: torch.Tensor, adj: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, adj)
        return self.head(h, pairs)


class GatedFusionCNNGAT(nn.Module):
    """Proposed v2 (conference novelty): PARALLEL temporal-CNN + GAT branches
    with learnable gated cross-attention fusion (not sequential, not concat).
    gate = sigmoid(W[cnn; gat; cnn-gat]); h = gate*cnn + (1-gate)*gat_attended."""

    def __init__(self, in_features: int, hidden: int = 48, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.cnn = TemporalCNN(in_features, hidden)
        self.static_proj = nn.Sequential(nn.Linear(in_features, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gat1 = GATLayer(hidden, hidden // heads, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.norm2 = nn.LayerNorm(hidden)
        self.head = PairHead(hidden)

    def encode(self, x, adj):
        b, n, t, f = x.shape
        hc = self.cnn(x)
        hs = self.static_proj(x[:, :, -1, :])
        hg = self.norm1(hs + F.elu(self.gat1(hs, adj)))
        g = self.gate(torch.cat([hc, hg, hc - hg], dim=-1))
        return self.norm2(g * hc + (1 - g) * hg)

    def forward(self, x, adj, pairs):
        return self.head(self.encode(x, adj), pairs)


class TransformerPair(nn.Module):
    """Transformer-over-history baseline (SOTA comparison)."""

    def __init__(self, in_features: int, hidden: int = 48, heads: int = 4):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=hidden * 2, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = PairHead(hidden)

    def forward(self, x, adj, pairs):
        b, n, t, f = x.shape
        h = self.proj(x.reshape(b * n, t, f))
        h = self.enc(h)[:, -1].view(b, n, -1)
        return self.head(h, pairs)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        deg = adj.sum(-1, keepdim=True).clamp(min=1.0)
        return self.lin(adj @ x / deg)


class GCNBaseline(nn.Module):
    """GCN baseline for graph-ablation completeness."""

    def __init__(self, in_features: int, hidden: int = 48):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden)
        self.g1 = GCNLayer(hidden, hidden)
        self.g2 = GCNLayer(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.head = PairHead(hidden)

    def forward(self, x, adj, pairs):
        h = x[:, :, -1, :]
        h = self.proj(h)
        h = self.norm(h + F.gelu(self.g2(F.gelu(self.g1(h, adj)), adj)))
        return self.head(h, pairs)


class CNNOnly(nn.Module):
    def __init__(self, in_features: int, hidden: int = 48):
        super().__init__()
        self.cnn = TemporalCNN(in_features, hidden)
        self.head = PairHead(hidden)

    def forward(self, x, adj, pairs):
        return self.head(self.cnn(x), pairs)


class GATOnly(nn.Module):
    def __init__(self, in_features: int, seq_len: int, hidden: int = 48, heads: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_features * seq_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.gat1 = GATLayer(hidden, hidden // heads, heads=heads)
        self.norm = nn.LayerNorm(hidden)
        self.head = PairHead(hidden)

    def forward(self, x, adj, pairs):
        b, n, t, f = x.shape
        h = self.proj(x.reshape(b, n, t * f))
        h = self.norm(h + F.elu(self.gat1(h, adj)))
        return self.head(h, pairs)


class LSTMModel(nn.Module):
    def __init__(self, in_features: int, hidden: int = 48):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = PairHead(hidden)

    def forward(self, x, adj, pairs):
        b, n, t, f = x.shape
        h, _ = self.lstm(x.reshape(b * n, t, f))
        h = h[:, -1].view(b, n, -1)
        return self.head(h, pairs)


class CNNLSTM(nn.Module):
    def __init__(self, in_features: int, hidden: int = 48):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_features, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = PairHead(hidden)

    def forward(self, x, adj, pairs):
        b, n, t, f = x.shape
        z = x.reshape(b * n, t, f).transpose(1, 2)
        z = self.conv(z).transpose(1, 2)
        h, _ = self.lstm(z)
        h = h[:, -1].view(b, n, -1)
        return self.head(h, pairs)


def build_model(name: str, in_features: int, seq_len: int, hidden: int = 48) -> nn.Module:
    name = name.lower()
    if name == "cnn_gat":
        return CNNGAT(in_features, hidden)
    if name == "cnn_gat_v2":
        return GatedFusionCNNGAT(in_features, hidden)
    if name == "transformer":
        return TransformerPair(in_features, hidden)
    if name == "gcn":
        return GCNBaseline(in_features, hidden)
    if name == "cnn":
        return CNNOnly(in_features, hidden)
    if name == "gat":
        return GATOnly(in_features, seq_len, hidden)
    if name == "lstm":
        return LSTMModel(in_features, hidden)
    if name == "cnn_lstm":
        return CNNLSTM(in_features, hidden)
    raise ValueError(f"unknown model {name}")
