"""TransHLA reimplementation — kiến trúc theo mô tả trong bài báo (GigaScience 2025).

Transformer encoder (trên sequence embedding ESM2)
  + Residual CNN nhánh 1 (trên sequence embedding)
  + Residual CNN nhánh 2 (trên contact map)
  -> nối lại -> bộ phân loại nhị phân.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# CNN Module 1 / CNN Module 2  (kiến trúc kiểu DPCNN, mục "The CNN module")
# --------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Khối tích chập đẳng kích thước: 2 lớp conv, mỗi lớp 256 feature map."""

    def __init__(self, channels=256, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)

    def forward(self, x):
        h = self.conv1(F.relu(x))
        h = self.conv2(F.relu(h))
        return h


class CNNModule1(nn.Module):
    """Region embedding -> conv block (có kết nối tắt)."""

    def __init__(self, in_dim, channels=256, kernel=3):
        super().__init__()
        self.region_embed = nn.Conv1d(in_dim, channels, kernel, padding=kernel // 2)
        self.block = ConvBlock(channels, kernel)

    def forward(self, x):            # x: (B, in_dim, L)
        x = self.region_embed(x)     # (B, 256, L)
        return x + self.block(x)     # kết nối tắt chống tiêu biến gradient


class CNNModule2(nn.Module):
    """Pooling giảm một nửa độ dài -> conv block -> kết nối tắt."""

    def __init__(self, channels=256, kernel=3):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block = ConvBlock(channels, kernel)

    def forward(self, x):
        x = self.pool(x)
        return x + self.block(x)


class ResidualCNN(nn.Module):
    """CNN Module 1 nối tiếp L lần CNN Module 2, kết thúc bằng BatchNorm."""

    def __init__(self, in_dim, seq_len, channels=256, n_blocks=None):
        super().__init__()
        self.m1 = CNNModule1(in_dim, channels)
        # lặp downsample cho tới khi độ dài còn 1
        if n_blocks is None:
            n_blocks, L = 0, seq_len
            while L > 1:
                L = (L + 1) // 2
                n_blocks += 1
        self.blocks = nn.ModuleList([CNNModule2(channels) for _ in range(n_blocks)])
        self.bn = nn.BatchNorm1d(channels)
        self.out_dim = channels

    def forward(self, x):            # (B, in_dim, L)
        x = self.m1(x)
        for b in self.blocks:
            x = b(x)
        x = self.bn(x)
        return x.mean(dim=-1)        # (B, 256)


# --------------------------------------------------------------------------
# Nhánh Transformer
# --------------------------------------------------------------------------
class TransformerBranch(nn.Module):
    def __init__(self, d_model=1280, nhead=8, n_layers=1, dim_ff=1280, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_dim = d_model

    def forward(self, x):            # (B, L, 1280)
        return self.encoder(x).mean(dim=1)


# --------------------------------------------------------------------------
# TransHLA
# --------------------------------------------------------------------------
class TransHLA(nn.Module):
    def __init__(self, seq_len, emb_dim=1280, channels=256,
                 nhead=8, n_tf_layers=1, dropout=0.3, n_classes=2):
        super().__init__()
        self.transformer = TransformerBranch(emb_dim, nhead, n_tf_layers, emb_dim, 0.1)
        self.cnn_seq = ResidualCNN(emb_dim, seq_len, channels)   # trên embedding
        self.cnn_con = ResidualCNN(seq_len, seq_len, channels)   # trên contact map
        fused = self.transformer.out_dim + self.cnn_seq.out_dim + self.cnn_con.out_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, 256), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, emb, contact, return_repr=False):
        """emb: (B, L, 1280)   contact: (B, L, L)"""
        t = self.transformer(emb)                  # (B, 1280)
        c1 = self.cnn_seq(emb.transpose(1, 2))     # (B, 256)
        c2 = self.cnn_con(contact)                 # (B, 256) — coi L hàng là L kênh
        rep = torch.cat([t, c1, c2], dim=-1)
        logits = self.head(rep)
        return (logits, rep) if return_repr else logits


# --------------------------------------------------------------------------
# TIM loss  (công thức 10–11 trong bài báo)
#   L = CE - H(Y) + alpha * H(Y|X),  alpha = 0.04
# --------------------------------------------------------------------------
class TIMLoss(nn.Module):
    def __init__(self, alpha=0.04, eps=1e-8):
        super().__init__()
        self.alpha, self.eps = alpha, eps

    def forward(self, logits, target):
        p = F.softmax(logits, dim=-1)                       # (B, K)
        ce = F.cross_entropy(logits, target)                # CE
        p_marg = p.mean(dim=0)                              # \hat p_k
        h_y = -(p_marg * torch.log(p_marg + self.eps)).sum()          # H(Y)
        h_y_x = -(p * torch.log(p + self.eps)).sum(dim=1).mean()      # H(Y|X)
        return ce - h_y + self.alpha * h_y_x, {
            'ce': ce.item(), 'H(Y)': h_y.item(), 'H(Y|X)': h_y_x.item()
        }


# --------------------------------------------------------------------------
# Các mô hình đối chứng (Bảng 2) — cùng đầu vào embedding ESM2
# --------------------------------------------------------------------------
class TextCNN(nn.Module):
    def __init__(self, seq_len, emb_dim=1280, n_filters=256, sizes=(2, 3, 4), n_classes=2):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in sizes])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(n_filters * len(sizes), n_classes)

    def forward(self, emb, contact=None):
        x = emb.transpose(1, 2)
        h = torch.cat([F.relu(c(x)).max(dim=-1).values for c in self.convs], dim=-1)
        return self.fc(self.drop(h))


class TextRCNN(nn.Module):
    def __init__(self, seq_len, emb_dim=1280, hidden=256, n_classes=2):
        super().__init__()
        self.rnn = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden * 2 + emb_dim, 256)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, emb, contact=None):
        h, _ = self.rnn(emb)
        h = torch.cat([h, emb], dim=-1)
        h = torch.tanh(self.fc1(h)).max(dim=1).values
        return self.fc2(self.drop(h))


class DPCNN(nn.Module):
    def __init__(self, seq_len, emb_dim=1280, channels=256, n_classes=2):
        super().__init__()
        self.body = ResidualCNN(emb_dim, seq_len, channels)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(channels, n_classes)

    def forward(self, emb, contact=None):
        return self.fc(self.drop(self.body(emb.transpose(1, 2))))


class RNNAtt(nn.Module):
    def __init__(self, seq_len, emb_dim=1280, hidden=256, n_classes=2):
        super().__init__()
        self.rnn = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(hidden * 2, 1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, emb, contact=None):
        h, _ = self.rnn(emb)
        w = torch.softmax(self.att(torch.tanh(h)), dim=1)
        return self.fc(self.drop((h * w).sum(dim=1)))


MODELS = {
    'TransHLA': TransHLA,
    'TextCNN': TextCNN,
    'TextRCNN': TextRCNN,
    'DPCNN': DPCNN,
    'RNN-ATTs': RNNAtt,
}
