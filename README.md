# Tái hiện TransHLA trên Google Colab

Bài báo: **TransHLA: a Hybrid Transformer model for HLA-presented epitope detection**
Lu T., Wang X., Nie W., Huo M., Li S. — *GigaScience* 14 (2025), DOI `10.1093/gigascience/giaf008`
Code gốc: `github.com/SkywalkerLuke/TransHLA` · Trọng số: `huggingface.co/SkywalkerLu/TransHLA_I` và `TransHLA_II`

**Quy trình VS Code + Colab:** xem [SETUP.md](SETUP.md).

---

## Ba notebook

| # | Notebook | GPU | Thời gian |
|---|---|---|---|
| 1 | `01_collect_data.ipynb` — tải và chuẩn hoá dữ liệu | không | 15–40 phút |
| 2 | `02_process_data.ipynb` — ESM2-650M embedding + contact map | **có** | 10 phút – 1,5 giờ |
| 3 | `03_run_experiment.ipynb` — huấn luyện, đánh giá, case study | **có** | 1,5 – 8 giờ |

Tất cả dùng chung thư mục Drive `MyDrive/s1156052/EpitopePrediction/`, nên có thể chạy từng phase ở các
phiên Colab khác nhau mà không mất dữ liệu.

```
EpitopePrediction/
├── data/          # Phase 1: hlaI_all.parquet, hlaII_all.parquet
├── cache/         # Phase 2: emb.f16, contact.f16, index.parquet, meta.json
├── checkpoints/   # Phase 3: *.pt
└── results/       # Phase 3: table2_*.csv, fig3_*.png, summary_*.json
```

---

## Thời gian chạy — trả lời ngắn

**Không lâu như vẻ ngoài của bài báo.** Phần nặng nhất (ESM2-650M) chỉ chạy **một lần** và
peptide rất ngắn (8–21 acid amin) nên rẻ bất ngờ. Phần huấn luyện thì mô hình chỉ ~15M tham số.

### Chạy trên tập con 150.000 peptide (mặc định) — Colab free T4

| bước | thời gian |
|---|---|
| Phase 1 | ~20 phút (chủ yếu là tải dữ liệu) |
| Phase 2 (mỗi lớp HLA) | ~5–10 phút |
| Phase 3 — TransHLA, 20 epoch | ~40–60 phút |
| Phase 3 — thêm 4 mô hình đối chứng | ~2–3 giờ |
| **Tổng cho cả HLA-I và HLA-II** | **≈ 6–8 giờ** |

### Chạy toàn bộ dữ liệu (919k + 624k peptide) — Colab free T4

| bước | thời gian |
|---|---|
| Phase 2, HLA-I | 30–50 phút (ghi 38 GB xuống đĩa) |
| Phase 2, HLA-II | 25–40 phút (38 GB) |
| Phase 3, TransHLA HLA-I, 20 epoch | 3–5 giờ |
| Phase 3, TransHLA HLA-II, 20 epoch | 2–4 giờ |
| 4 mô hình đối chứng × 2 lớp | 8–14 giờ |
| **Tổng** | **≈ 20–30 giờ GPU** |

Trên **A100** (Colab Pro+) con số này rút còn **5–8 giờ**.

### Điều đáng lo hơn thời gian: dung lượng

Embedding đầy đủ chiếm **~38 GB mỗi lớp HLA** (fp16). Google Drive miễn phí có 15 GB.
Vì vậy notebook mặc định đặt `SUBSET_N = 150_000` (~6 GB). Muốn chạy đầy đủ thì:

* đặt `CACHE_ON_DRIVE = False` (ghi vào đĩa cục bộ ~100 GB của Colab) **và chạy Phase 2 + Phase 3 trong cùng một phiên**, hoặc
* dùng Colab Pro + Drive trả phí.

### Phần **không** chạy được trên Colab

Quy trình sinh negative samples của bài báo cần BLAST (DIAMOND) toàn bộ peptide dương tính
ngược lại **NCBI nr** (~400 GB) rồi khử dư thừa bằng CD-HIT. Việc này cần máy chủ riêng,
≥500 GB đĩa và **1–2 ngày**. Notebook 1 vì vậy mặc định dùng bộ dữ liệu train/val/test mà
tác giả đã công bố — đây là cách tái hiện đúng và khả thi. Pipeline đầy đủ vẫn được ghi lại
ở cuối notebook 1 (`OPTION B`) nếu bạn có hạ tầng phù hợp.

---

## Thứ tự chạy

```
01_collect_data.ipynb                 (CPU, chạy 1 lần)
02_process_data.ipynb  HLA_CLASS='I'  (GPU)
03_run_experiment.ipynb HLA_CLASS='I' (GPU)
02_process_data.ipynb  HLA_CLASS='II' (GPU)
03_run_experiment.ipynb HLA_CLASS='II'(GPU)
```

Notebook 2 có cơ chế **resume**: nếu Colab ngắt giữa chừng, chạy lại là nó tiếp tục từ chỗ dừng.
Notebook 3 lưu checkpoint tốt nhất sau mỗi epoch vào Drive.

---

## Kết quả cần khớp

**Bảng 2** — tập test IEDB (92.347 epitope HLA-I, 65.105 epitope HLA-II, 157.879 chuỗi ngẫu nhiên):

| | ACC | F1 | Recall | MCC | AUC |
|---|---|---|---|---|---|
| TransHLA, HLA-I | 84,72 | 84,59 | 83,92 | 0,69 | 91,95 |
| TransHLA, HLA-II | 79,94 | 81,07 | 86,19 | 0,60 | 88,14 |

Mô hình đứng thứ hai (DPCNN) đạt 83,75 / 90,97 ở HLA-I và 77,41 / 85,30 ở HLA-II.
Trên tập neoepitope NEPDB đã kiểm chứng bằng thực nghiệm TCR, TransHLA đạt **90,65%** độ chính xác.

Notebook 3 in sẵn bảng so sánh *tái hiện vs bài báo* và có thêm mục đối chiếu với **trọng số
chính thức trên HuggingFace** chạy trên đúng tập test của bạn — cách nhanh nhất để biết sai lệch
đến từ dữ liệu hay từ quá trình huấn luyện.

---

## Những chỗ bài báo không nêu rõ (đã chọn giá trị hợp lý)

| tham số | bài báo | notebook dùng |
|---|---|---|
| số epoch | không nêu | 20 + early stopping theo AUC |
| learning rate / optimizer | không nêu | AdamW, 1e-4, cosine schedule |
| batch size | không nêu | 256 |
| số lớp Transformer encoder | không nêu | 1 (đổi qua `n_tf_layers`) |
| số head | không nêu | 8 |
| α của TIM loss | **0,04** | 0,04 |
| lớp biểu diễn ESM2 | không nêu | layer 33 (cuối) |
| mô hình ESM2 | ESM2 | `esm2_t33_650M_UR50D` (theo model card HF) |
| độ dài đệm | đệm cuối bằng 1 | 16 (HLA-I) / 23 (HLA-II), theo model card HF |

Nếu kết quả lệch, notebook 3 có danh sách kiểm tra theo thứ tự ưu tiên ở ô cuối cùng.
