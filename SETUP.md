# Quy trình làm việc: sửa code ở VS Code — chạy trên Google Colab

Repo: <https://github.com/Tienanh209/TransHLA_repro> (công khai)

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

## 2. Chuẩn bị trên Colab

Không cần gì cả — repo công khai nên ô bootstrap clone được ngay.

> Nếu sau này chuyển repo về riêng tư (`gh repo edit --visibility private`), tạo
> **fine-grained token** tại <https://github.com/settings/tokens> (chọn repo `TransHLA_repro`,
> quyền **Contents: Read-only**) rồi lưu trong Colab ở biểu tượng **chìa khoá 🔑** → **Add new
> secret**, tên `GH_TOKEN`, bật **Notebook access**. Ô bootstrap tự nhận ra và dùng.

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

## 5. Thứ tự chạy

Giai đoạn 1 chỉ có một notebook, chạy từ đầu đến cuối là xong cả hai lớp HLA:

```
01_eval_pretrained.ipynb   (GPU)
```

Không cần Drive — dữ liệu clone trực tiếp từ repo gốc của tác giả. Xem `README.md` cho thời
gian chạy và kết quả cần khớp.

> `legacy_scaffold/` là bản thử đầu tiên, không dùng để tái hiện. `README.md` giải thích lý do.
