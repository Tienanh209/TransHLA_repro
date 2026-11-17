# Tái hiện TransHLA

Bài báo: **TransHLA: a Hybrid Transformer model for HLA-presented epitope detection**
Lu T., Wang X., Nie W., Huo M., Li S. — *GigaScience* 14 (2025), DOI `10.1093/gigascience/giaf008`

| | |
|---|---|
| Code gốc | `github.com/SkywalkerLuke/TransHLA` |
| Trọng số | `huggingface.co/SkywalkerLu/TransHLA_I` và `TransHLA_II` |
| Quy trình làm việc | VS Code (Mac) → GitHub → Colab, xem [SETUP.md](SETUP.md) |

---

## Tác giả công bố nhiều hơn bài báo viết

Đây là điều quan trọng nhất khi bắt đầu, vì nó rút ngắn công việc đi rất nhiều:

| | trong repo gốc |
|---|---|
| **Dữ liệu** | **có sẵn toàn bộ** tập train/validation/test đã chia, cộng NEPDB và external 1×/4× |
| **Model** | `model_train_test/model.py` có `TransHLA` **và cả 4 baseline** (TextCNN, TextRCNN, DPCNN, RNN_ATTs) |
| **Siêu tham số** | `train.py`: Adam `lr=3e-5`, `weight_decay=0.0025`, batch 64, 100 epoch, early stopping patience 5, seed 10038, `n_layers=6`, `n_head=8`, `d_model=1280`, `dim_feedforward=64` |
| **Embedding** | ESM2 `esm2_t33_650M_UR50D`, biểu diễn layer 33 + contact map (`return_contacts=True`), **đóng băng** (`torch.no_grad`) |

Số lượng mẫu trong tập test khớp đúng Bảng 2: **92.347** epitope HLA-I dương tính và **65.105**
epitope HLA-II dương tính. Nghĩa là **không cần** dựng lại quy trình sinh mẫu âm của bài báo
(BLAST toàn bộ peptide dương tính ngược NCBI nr ~400 GB rồi khử dư thừa bằng CD-HIT) — việc
cần máy chủ riêng và 1-2 ngày.

---

## Trạng thái

| giai đoạn | tệp | trạng thái |
|---|---|---|
| **1. Đánh giá trọng số đã công bố** | [`01_eval_pretrained.ipynb`](01_eval_pretrained.ipynb) · [`transhla_eval.py`](transhla_eval.py) | xong, chưa chạy |
| 2. Huấn luyện lại từ đầu | dùng `train.py` gốc | chưa làm |
| 3. Huấn luyện 4 baseline | dùng `model.py` gốc | chưa làm |
| 4. Case study CEDAR / VDJdb | — | chưa làm |

Giai đoạn 1 là bước nên làm trước tiên: nếu con số khớp Bảng 2, ta biết dữ liệu và quy trình
đánh giá đã đúng, nên mọi sai lệch về sau chỉ có thể đến từ quá trình huấn luyện.

---

## Chạy giai đoạn 1

Mở `01_eval_pretrained.ipynb` trên Colab (File → Open notebook → GitHub →
`Tienanh209/TransHLA_repro`), chọn **Runtime → GPU**, rồi Run all.

* không cần Google Drive — dữ liệu clone trực tiếp từ repo gốc (~250 MB)
* tải về ESM2-650M (~2,6 GB) và 2 checkpoint TransHLA (~2,6 GB mỗi cái)
* ~10-25 phút cho cả hai lớp HLA trên A100

Hoặc chạy bằng dòng lệnh trên máy có GPU:

```bash
python transhla_eval.py --hla both --dataset test --out results/
```

---

## Kết quả cần khớp

**Bảng 2** — tập test IEDB (184.697 peptide HLA-I, 130.634 peptide HLA-II):

| | ACC (%) | F1 (%) | Recall (%) | MCC | AUC (%) |
|---|---|---|---|---|---|
| TransHLA, HLA-I | 84,72 | 84,59 | 83,92 | 0,69 | 91,95 |
| TransHLA, HLA-II | 79,94 | 81,07 | 86,19 | 0,60 | 88,14 |

Mô hình đứng thứ hai (DPCNN) đạt 83,75 / 90,97 ở HLA-I và 77,41 / 85,30 ở HLA-II.
Số đầy đủ của cả 4 baseline nằm trong `transhla_eval.PAPER_BASELINES_TABLE2`.

**Bảng 4** — NEPDB, neoepitope đã kiểm chứng bằng thí nghiệm TCR: ACC **90,65%**,
recall **93,52%**, specificity **87,77%**.

Notebook in sẵn bảng *tái hiện vs bài báo* kèm độ lệch cho từng chỉ số.

---

## Những chỗ dễ sai

Đã xử lý sẵn trong `transhla_eval.py`, ghi lại đây để khỏi vấp lại:

1. **Độ dài đệm.** HLA-I đệm tới **16** token, HLA-II tới **23** (= `max_len` 14/21 cộng `<cls>`
   và `<eos>`), đệm bằng token **1** (`<pad>`). Sai thì contact map của ESM2 lệch chiều.
2. **Đầu ra đã là softmax.** `model(tokens)` trả `(probs, representation)`, `probs[:, 1]` là
   p(epitope). Đừng softmax lần nữa.
3. **Không lọc tập test.** Không khử trùng lặp (`train.py` chỉ khử ở tập train), không loại
   peptide chứa `X` (ESM2 có ký tự này trong từ điển). Loại đi là mất 169 dòng âm của HLA-I.
4. **Phải lọc độ dài tập external.** Các tập này chứa peptide 7-29 acid amin; hàm đệm của tác
   giả chỉ đệm chứ không cắt nên sẽ lỗi tensor lệch chiều.
5. **`model.eval()`.** Mô hình có BatchNorm và Dropout.
6. **Peptide 13-14 acid amin** nằm trong khoảng của cả hai mô hình — đừng lẫn mô hình nọ với
   tập kia.

---

## Giai đoạn 2 — huấn luyện lại

Siêu tham số đã có đủ nên chạy trực tiếp `train.py` của tác giả:

```bash
cd TransHLA_official/model_train_test
python train.py --train_path ../data/HLA_I_epitope_train_shuffle.csv \
                --validation_path ../data/HLA_I_epitope_validation.csv \
                --model_path ./checkpoints --model_name transhla_I.pt
```

Hai điều cần biết trước:

* `train.py` gọi ESM2 trong `forward` **mỗi batch, mỗi epoch**, tính lại từ đầu. Cache
  embedding + contact map một lần sẽ nhanh hơn nhiều, đổi lại ~38 GB mỗi lớp HLA ở fp16.
* `max_len` mặc định là 14 (HLA-I). Chạy HLA-II phải truyền `-max_len 21` — chính tác giả ghi
  chú việc này trong `train.py`.

---

## `legacy_scaffold/`

Bản thử đầu tiên: 3 notebook tự viết lại mô hình từ bài báo, kèm quy trình cache embedding
xuống Drive. Giữ lại để tham khảo nhưng **không dùng để tái hiện** — README của nó ghi rằng
bài báo không nêu epoch/lr/batch rồi tự chọn (AdamW 1e-4, batch 256, 20 epoch, 1 lớp encoder),
trong khi code gốc có đủ và khác hẳn (Adam 3e-5, batch 64, 6 lớp). Huấn luyện theo đó sẽ không
ra số của bài báo.
