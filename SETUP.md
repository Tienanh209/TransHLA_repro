# Quy trình làm việc: sửa code ở VS Code — chạy trên Google Colab

Repo: <https://github.com/Tienanh209/TransHLA_repro> (riêng tư)

```
VS Code (Mac)  --- git push --->  GitHub  --- git pull --->  Colab (GPU)
   sửa code                                                  chạy, train
```

GitHub là nguồn sự thật cho **code**. Google Drive (`MyDrive/s1156052/EpitopePrediction/`)
vẫn là nơi chứa **dữ liệu, cache, checkpoint, kết quả** — những thứ này không đẩy lên git.

---

## 1. Chuẩn bị một lần trên máy

```bash
cd /Users/macos/Documents/biolab/epitode/TransHLA_repro && code .
```

Khi VS Code hỏi, bấm **Install** cho các extension gợi ý (Python, Jupyter, Pylance, Ruff).

Môi trường ảo để có autocomplete (không bắt buộc, không dùng để train):

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Sau đó trong VS Code: `Cmd+Shift+P` → **Python: Select Interpreter** → chọn `.venv`.

## 2. Chuẩn bị một lần trên Colab

Repo ở chế độ riêng tư nên Colab cần token để clone:

1. Tạo token tại <https://github.com/settings/tokens> → **Fine-grained token**, chọn repo
   `TransHLA_repro`, quyền **Contents: Read-only**.
2. Trong Colab, bấm biểu tượng **chìa khoá 🔑** ở cột trái → **Add new secret**
   → Name `GH_TOKEN`, Value là token vừa tạo → bật **Notebook access**.

Token chỉ cần làm một lần, dùng chung cho cả ba notebook.

> Nếu thấy phiền, có thể để repo ở chế độ công khai (`gh repo edit --visibility public`)
> rồi bỏ qua bước token — ô bootstrap vẫn chạy bình thường khi không có `GH_TOKEN`.

## 3. Vòng lặp hằng ngày

**Trên máy (VS Code)** — sửa `models_transhla.py` hoặc notebook, rồi:

```bash
git add -A && git commit -m "sửa gì đó" && git push
```

**Trên Colab** — mở notebook (File → Open notebook → GitHub → `Tienanh209/TransHLA_repro`),
chạy ô **"0. Đồng bộ code từ GitHub"** ở đầu notebook. Ô này clone hoặc `reset --hard` về đúng
bản trên GitHub, `cd` vào repo và thêm nó vào `sys.path`, nên `from models_transhla import ...`
lấy đúng code bạn vừa push.

Sửa `.py` giữa chừng mà không muốn restart runtime:

```python
import importlib, models_transhla
importlib.reload(models_transhla)
from models_transhla import TransHLA, TIMLoss, MODELS
```

> Ô bootstrap **ghi đè** mọi thay đổi gõ trực tiếp trên Colab. Muốn giữ gì thì copy về máy
> và commit ở đó.

## 4. Notebook sửa ở đâu?

Hai cách, chọn một để tránh xung đột:

* **Sửa trên máy** (khuyến nghị): mở `.ipynb` trong VS Code, chỉ sửa code — đừng chạy ở máy
  (không có GPU, không có Drive). Push rồi mở bản trên Colab bằng File → Open notebook → GitHub.
* **Sửa trên Colab**: dùng File → Save a copy in GitHub để commit ngược lại repo, rồi
  `git pull` ở máy.

Đừng làm cả hai cùng lúc trên cùng một notebook.

## 5. Thứ tự chạy (không đổi)

```
01_collect_data.ipynb                  (CPU)
02_process_data.ipynb   HLA_CLASS='I'  (GPU)
03_run_experiment.ipynb HLA_CLASS='I'  (GPU)
02_process_data.ipynb   HLA_CLASS='II' (GPU)
03_run_experiment.ipynb HLA_CLASS='II' (GPU)
```

Xem `README.md` cho thời gian chạy, dung lượng và kết quả cần khớp.
