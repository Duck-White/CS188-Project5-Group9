# Website Project 5 — CS188 Fall 2025 — Nhóm 9

Website được thiết kế lại **chỉ cho Q1–Q6**, bám sát đề gốc Berkeley:
<https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/>.

**Quan trọng:** Berkeley không yêu cầu nộp website. Ba file được chấm chính thức là `models.py`, `train.py`, `losses.py`. Website là công cụ minh họa và thực nghiệm, **không thay thế** `autograder.py`.

## Cách chạy website

Mở terminal trong thư mục có `models.py`, `train.py` và `webapp/`:

```bash
python -m pip install -r requirements-web.txt
python -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000
```

Mở trình duyệt tại **http://127.0.0.1:8000**. Không mở `docs/index.html` bằng cách nhấp đúp nếu muốn chạy PyTorch: khi chỉ mở HTML hoặc đăng lên GitHub Pages, giao diện vẫn hiển thị nhưng **không thể gọi các mô hình Python**.

## Các thao tác đã kết nối với mô hình thật

| Câu | Chức năng website | Cơ chế thực |
|---|---|---|
| Q1 | Chạy Perceptron và quan sát đường phân loại | `PerceptronModel` + `train_perceptron` trên 12 điểm separable minh họa |
| Q2 | Huấn luyện hồi quy sin(x), vẽ đồ thị, tính MSE | `RegressionModel` + `train_regression` trên 200 điểm mẫu |
| Q3 | Vẽ chữ số 28×28, nhận dạng MNIST | `DigitClassificationModel`, **chỉ khi có `q3.pt`** |
| Q4 | Nhập từ, dự đoán 5 ngôn ngữ | `LanguageIDModel`, **chỉ khi có `q4.pt` và bảng chữ cái** |
| Q5 | Kiểm chứng phép Convolve với ma trận, vẽ số cho CNN | `models.Convolve`; phân loại **chỉ khi có `q5.pt`** |
| Q6 | Hiển thị causal attention weights và đầu ra | `Attention.forward` với trọng số ngẫu nhiên seed 188, không nhận là model đã học |

**Không có checkpoint giả.** Q3/Q4/Q5 trả về thông báo rõ ràng nếu thiếu trọng số.

## Xuất checkpoint sau khi huấn luyện

Hãy đặt các file dữ liệu Berkeley theo đường dẫn `data/mnist.npz`, `data/lang_id.npz` (tải bộ khung chính thức).
Sau đó có thể chạy:

```bash
python -m webapp.export_checkpoints --question q3
python -m webapp.export_checkpoints --question q4
python -m webapp.export_checkpoints --question q5
```

Script này huấn luyện bằng các hàm trong `train.py` rồi lưu kết quả vào thư mục `checkpoints/`. Q4 còn xuất `q4_chars.json` để bảo đảm thứ tự 47 ký tự giống file dữ liệu. Mỗi lần huấn luyện có thể tốn tài nguyên; chỉ dùng khi cần thực nghiệm trên web. Checkpoint không nằm trong ZIP, và việc tạo checkpoint **không tự động xác nhận đạt điểm Berkeley**.

## Chấm điểm chính thức

```bash
python autograder.py --no-graphics
# Hoặc
python autograder.py -q q5 --no-graphics
```

Bản trước tối ưu Q5 đã có nhật ký hai seed 25/25. Bản code tối ưu được chấm lại Q1 (5/5), Q2 (5/5), Q6 (2/2) và smoke tests; chưa đủ dữ liệu để xác nhận tổng 25/25 trên bản tối ưu. Khi có `data/`, chấm lại Q3–Q5 như hướng dẫn.

## Kiểm thử website

```bash
python -m pip install pytest httpx
python -m pytest -q tests/test_web_api.py
```

Bộ kiểm thử này chạy qua PyTorch thật cho Q1, Q2, Q5-Convolve, Q6, kiểm tra việc từ chối dự đoán Q3/Q4/Q5 nếu thiếu checkpoint. Bộ kiểm thử **không thay thế autograder của Berkeley**.

## Cấu trúc

```text
models.py / train.py / losses.py   # Bài nộp Q1–Q6
backend.py / autograder.py        # Công cụ gốc, giữ nguyên
webapp/server.py                 # REST API FastAPI ↔ PyTorch
webapp/export_checkpoints.py     # Train trên dữ liệu Berkeley và lưu trọng số
docs/index.html                  # Trang web
docs/assets/style.css            # Giao diện responsive
docs/assets/app.js               # Tương tác, biểu đồ, canvas
checkpoints/                      # q3.pt, q4.pt, q5.pt (nếu tạo)
requirements-web.txt             # Thư viện chạy web
```

Không có Question 7.