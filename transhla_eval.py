"""Đánh giá trọng số TransHLA đã công bố trên đúng tập test của bài báo.

Bài báo: Lu T., Wang X., Nie W., Huo M., Li S. (2025) "TransHLA: a Hybrid Transformer
model for HLA-presented epitope detection", GigaScience 14, DOI 10.1093/gigascience/giaf008

Mọi thứ ở đây đi theo code gốc của tác giả (github.com/SkywalkerLuke/TransHLA):
  - tokenizer: facebook/esm2_t33_650M_UR50D
  - đệm bằng token 1 (<pad>) tới 16 token cho HLA-I, 23 token cho HLA-II
    (= max_len 14 / 21 cộng <cls> và <eos>)
  - model: SkywalkerLu/TransHLA_I, SkywalkerLu/TransHLA_II (trust_remote_code)
  - đầu ra cột 1 đã là softmax p(epitope); nhãn dự đoán = argmax
    (xem model.py: `logits_clsf = F.softmax(logits_clsf, dim=1)`)

Dùng được cả trên Colab (import vào notebook) và dòng lệnh:
    python transhla_eval.py --hla I --dataset test --out results/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

OFFICIAL_REPO = "https://github.com/SkywalkerLuke/TransHLA.git"

# ---------------------------------------------------------------------------
# Cấu hình theo lớp HLA — lấy từ TransHLA_I.py / TransHLA_II.py của tác giả
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HLAConfig:
    hla: str
    hf_model: str
    pad_to: int  # độ dài token sau khi đệm
    min_len: int  # độ dài peptide hợp lệ (acid amin)
    max_len: int
    datasets: dict[str, str] = field(default_factory=dict)


CONFIGS = {
    "I": HLAConfig(
        hla="I",
        hf_model="SkywalkerLu/TransHLA_I",
        pad_to=16,
        min_len=8,
        max_len=14,
        datasets={
            "test": "HLA_I_epitope_test.csv",
            "validation": "HLA_I_epitope_validation.csv",
            "nepdb": "NEPDB_I.csv",
            "external_1x": "HLA_I_external_1_time_negative.csv",
            "external_4x": "HLA_I_external_4_time_negative.csv",
        },
    ),
    "II": HLAConfig(
        hla="II",
        hf_model="SkywalkerLu/TransHLA_II",
        pad_to=23,
        min_len=13,
        max_len=21,
        datasets={
            "test": "HLA_II_epitope_test.csv",
            "validation": "HLA_II_epitope_validation.csv",
            "nepdb": "NEPDB_II.csv",
            "external_1x": "HLA_II_external_1_time_negative.csv",
            "external_4x": "HLA_II_external_4_time_negative.csv",
        },
    ),
}

# Số của bài báo để đối chiếu.
# Bảng 2 (tập test IEDB) và Bảng 4 (NEPDB, epitope đã kiểm chứng bằng TCR).
PAPER = {
    ("I", "test"): {"acc": 84.72, "f1": 84.59, "recall": 83.92, "mcc": 0.69, "auc": 91.95},
    ("II", "test"): {"acc": 79.94, "f1": 81.07, "recall": 86.19, "mcc": 0.60, "auc": 88.14},
    ("I", "nepdb"): {"acc": 90.65, "recall": 93.52, "specificity": 87.77},
}

PAPER_BASELINES_TABLE2 = pd.DataFrame(
    [
        ("HLA-I", "TransHLA", 84.72, 84.59, 83.92, 0.69, 91.95),
        ("HLA-I", "TextCNN", 81.63, 79.61, 83.50, 0.63, 89.37),
        ("HLA-I", "TextRCNN", 81.21, 76.65, 83.78, 0.64, 87.62),
        ("HLA-I", "DPCNN", 83.75, 83.76, 83.96, 0.68, 90.97),
        ("HLA-I", "RNN-ATTs", 81.17, 82.27, 81.46, 0.63, 87.98),
        ("HLA-II", "TransHLA", 79.94, 81.07, 86.19, 0.60, 88.14),
        ("HLA-II", "TextCNN", 73.26, 73.49, 72.64, 0.47, 80.64),
        ("HLA-II", "TextRCNN", 70.96, 72.02, 69.28, 0.42, 78.21),
        ("HLA-II", "DPCNN", 77.41, 77.91, 75.98, 0.55, 85.30),
        ("HLA-II", "RNN-ATTs", 69.04, 69.83, 67.89, 0.38, 75.81),
    ],
    columns=["Type", "Method", "ACC (%)", "F1 (%)", "Recall (%)", "MCC", "AUC (%)"],
)


# ---------------------------------------------------------------------------
# Dữ liệu
# ---------------------------------------------------------------------------


def ensure_official_repo(dest: Path) -> Path:
    """Clone repo gốc của tác giả (chứa cả dữ liệu đã chia) nếu chưa có."""
    dest = Path(dest)
    if (dest / "data").is_dir():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", OFFICIAL_REPO, str(dest)],
        check=True,
    )
    return dest


# Từ điển ESM2 chấp nhận cả ký tự nhập nhằng (X = acid amin bất kỳ, B/Z/U/O),
# nên không loại chúng — loại đi là lệch khỏi tập test của tác giả.
ESM2_RESIDUES = "ACDEFGHIKLMNPQRSTVWYXBUZO"


def load_dataset(
    official_repo: Path,
    cfg: HLAConfig,
    name: str,
    subset_n: int | None = None,
    seed: int = 0,
    dedup: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Đọc một tập dữ liệu của tác giả, lọc theo khoảng độ dài của lớp HLA.

    Trả về (dataframe có cột `peptide`/`label`, thông tin về số dòng bị loại).

    Lọc độ dài là cần thiết: các tập external chứa peptide 7-29 acid amin, nằm
    ngoài khoảng mô hình được huấn luyện (8-14 cho HLA-I, 13-21 cho HLA-II), và
    hàm đệm của tác giả chỉ đệm chứ không cắt nên sẽ lỗi tensor lệch chiều.

    `dedup` mặc định False: `train.py` của tác giả chỉ `drop_duplicates` ở tập
    train, nên giữ nguyên tập test mới khớp đúng số dòng của Bảng 2.
    """
    path = Path(official_repo) / "data" / cfg.datasets[name]
    raw = pd.read_csv(path)
    df = raw.iloc[:, :2].copy()
    df.columns = ["peptide", "label"]
    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()
    df["label"] = df["label"].astype(int)

    n_raw = len(df)
    n_dedup = 0
    if dedup:
        df = df.drop_duplicates(subset="peptide", keep="first")
        n_dedup = n_raw - len(df)

    lengths = df["peptide"].str.len()
    in_range = lengths.between(cfg.min_len, cfg.max_len)
    valid_aa = df["peptide"].str.fullmatch(f"[{ESM2_RESIDUES}]+")
    keep = in_range & valid_aa
    info = {
        "file": cfg.datasets[name],
        "rows_raw": n_raw,
        "dropped_duplicate": int(n_dedup),
        "dropped_out_of_length": int((~in_range).sum()),
        "dropped_bad_residue": int((in_range & ~valid_aa).sum()),
        "n_duplicate_peptides": int(df["peptide"].duplicated().sum()),
        "n_ambiguous_residue": int(df["peptide"].str.contains("[XBUZO]").sum()),
    }
    df = df[keep].reset_index(drop=True)

    if subset_n is not None and len(df) > subset_n:
        df = df.sample(n=subset_n, random_state=seed).reset_index(drop=True)
        info["subset_n"] = subset_n

    info["rows_used"] = len(df)
    info["positives"] = int((df["label"] == 1).sum())
    info["negatives"] = int((df["label"] == 0).sum())
    return df, info


# ---------------------------------------------------------------------------
# Mô hình và suy luận
# ---------------------------------------------------------------------------


def pad_to_length(token_lists: list[list[int]], target_length: int) -> list[list[int]]:
    """Đệm bằng token 1 (<pad>) — y như `pad_inner_lists_to_length` của tác giả."""
    for inner in token_lists:
        if len(inner) > target_length:
            raise ValueError(
                f"chuỗi dài {len(inner)} token, vượt {target_length}; hãy lọc độ dài trước"
            )
        inner.extend([1] * (target_length - len(inner)))
    return token_lists


def load_model(cfg: HLAConfig, device: torch.device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModel.from_pretrained(cfg.hf_model, trust_remote_code=True)
    model.to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict(
    peptides: list[str],
    cfg: HLAConfig,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 128,
    amp: bool = False,
    progress: bool = True,
) -> np.ndarray:
    """Trả về p(epitope) cho từng peptide.

    `amp=True` bật autocast fp16 trên CUDA — nhanh gần gấp đôi nhưng con số có thể
    lệch ở chữ số thập phân thứ 3. Mặc định tắt để trung thành với bài báo.
    """
    encoding = tokenizer(list(peptides))["input_ids"]
    encoding = pad_to_length(encoding, cfg.pad_to)
    tokens = torch.tensor(encoding, dtype=torch.long)

    iterator = range(0, len(tokens), batch_size)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc=f"TransHLA_{cfg.hla}", unit="batch")
        except ImportError:
            pass

    probs = []
    for start in iterator:
        batch = tokens[start : start + batch_size].to(device)
        if amp and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out, _ = model(batch)
        else:
            out, _ = model(batch)
        probs.append(out[:, 1].float().cpu().numpy())
    return np.concatenate(probs)


# ---------------------------------------------------------------------------
# Chỉ số
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    """ACC/F1/Recall/Precision/Specificity/MCC/AUC — ngưỡng 0,5 = argmax của tác giả."""
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "acc": 100 * accuracy_score(y_true, y_pred),
        "f1": 100 * f1_score(y_true, y_pred, zero_division=0),
        "recall": 100 * recall_score(y_true, y_pred, zero_division=0),
        "precision": 100 * precision_score(y_true, y_pred, zero_division=0),
        "specificity": 100 * tn / (tn + fp) if (tn + fp) else float("nan"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc": 100 * roc_auc_score(y_true, prob),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def comparison_table(hla: str, dataset: str, got: dict) -> pd.DataFrame:
    """Bảng tái hiện vs bài báo, kèm độ lệch."""
    ref = PAPER.get((hla, dataset), {})
    rows = []
    labels = {
        "acc": "ACC (%)",
        "f1": "F1 (%)",
        "recall": "Recall (%)",
        "precision": "Precision (%)",
        "specificity": "Specificity (%)",
        "mcc": "MCC",
        "auc": "AUC (%)",
    }
    for key, label in labels.items():
        if key not in got:
            continue
        paper = ref.get(key)
        rows.append(
            {
                "Chỉ số": label,
                "Tái hiện": round(got[key], 2 if key != "mcc" else 3),
                "Bài báo": paper,
                "Lệch": round(got[key] - paper, 2) if paper is not None else None,
            }
        )
    return pd.DataFrame(rows)


def plot_curves(y_true: np.ndarray, prob: np.ndarray, title: str, out_path: Path | None = None):
    """ROC + PR, tương ứng Hình 3 của bài báo."""
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true, prob)
    auc_val = roc_auc_score(y_true, prob)
    prec, rec, _ = precision_recall_curve(y_true, prob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, lw=2, label=f"TransHLA (AUC = {auc_val:.4f})")
    axes[0].plot([0, 1], [0, 1], "--", lw=1, color="grey")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC — {title}")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec, lw=2)
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision–Recall — {title}")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Quy trình hoàn chỉnh cho một (lớp HLA, tập dữ liệu)
# ---------------------------------------------------------------------------


def run(
    hla: str,
    dataset: str = "test",
    official_repo: Path | str = "TransHLA_official",
    out_dir: Path | str | None = None,
    batch_size: int = 128,
    subset_n: int | None = None,
    amp: bool = False,
    device: torch.device | None = None,
    model_cache: dict | None = None,
    dedup: bool = False,
) -> dict:
    cfg = CONFIGS[hla]
    repo = ensure_official_repo(Path(official_repo))
    df, info = load_dataset(repo, cfg, dataset, subset_n=subset_n, dedup=dedup)

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    if model_cache is not None and hla in model_cache:
        tokenizer, model = model_cache[hla]
    else:
        tokenizer, model = load_model(cfg, device)
        if model_cache is not None:
            model_cache[hla] = (tokenizer, model)

    t0 = time.time()
    prob = predict(
        df["peptide"].tolist(), cfg, tokenizer, model, device,
        batch_size=batch_size, amp=amp,
    )
    elapsed = time.time() - t0

    metrics = compute_metrics(df["label"].to_numpy(), prob)
    result = {
        "hla": hla,
        "dataset": dataset,
        "hf_model": cfg.hf_model,
        "device": str(device),
        "amp": amp,
        "seconds": round(elapsed, 1),
        "peptides_per_second": round(len(df) / elapsed, 1) if elapsed else None,
        "data": info,
        "metrics": metrics,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"hla{hla}_{dataset}"
        pd.DataFrame(
            {"peptide": df["peptide"], "label": df["label"], "prob": prob}
        ).to_csv(out_dir / f"predictions_{tag}.csv", index=False)
        comparison_table(hla, dataset, metrics).to_csv(
            out_dir / f"comparison_{tag}.csv", index=False
        )
        (out_dir / f"summary_{tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        result["out_dir"] = str(out_dir)

    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hla", choices=["I", "II", "both"], default="both")
    p.add_argument("--dataset", default="test", help="test | validation | nepdb | external_1x | external_4x")
    p.add_argument("--official-repo", default="TransHLA_official")
    p.add_argument("--out", default="results")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--subset-n", type=int, default=None, help="chạy thử trên N peptide lấy mẫu ngẫu nhiên")
    p.add_argument("--amp", action="store_true", help="autocast fp16 trên CUDA (nhanh hơn, lệch nhẹ)")
    args = p.parse_args()

    classes = ["I", "II"] if args.hla == "both" else [args.hla]
    for hla in classes:
        res = run(
            hla,
            dataset=args.dataset,
            official_repo=args.official_repo,
            out_dir=args.out,
            batch_size=args.batch_size,
            subset_n=args.subset_n,
            amp=args.amp,
        )
        print(f"\n=== HLA-{hla} / {args.dataset} ===")
        print(json.dumps(res["data"], indent=2, ensure_ascii=False))
        print(comparison_table(hla, args.dataset, res["metrics"]).to_string(index=False))
        print(f"({res['seconds']}s, {res['peptides_per_second']} peptide/s trên {res['device']})")


if __name__ == "__main__":
    main()
