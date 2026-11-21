# Web demo Q1–Q6 — Project 5 Fall 2025

Đây là demo bổ sung cho Nhóm 9. Berkeley chấm `models.py`, `train.py`, `losses.py`; website không thay thế autograder.

## Chạy trên máy bằng PyTorch

Cài `pip install -r requirements-web.txt`, chạy `python -m uvicorn webapp.server:app --port 8000`, mở http://127.0.0.1:8000. Q1/Q2, Convolve Q5, Q6 hoạt động mà không cần checkpoint. Q3/Q4 và CNN Q5 cần checkpoint đã huấn luyện.

Đặt `data/mnist.npz`, `data/lang_id.npz` từ bộ dữ liệu Berkeley vào `data/`, rồi chạy:

```bash
python -m webapp.export_checkpoints --question q3
python -m webapp.export_checkpoints --question q4
python -m webapp.export_checkpoints --question q5
```

Script lưu `checkpoints/q3.pt`, `q4.pt`, `q4_chars.json`, `q5.pt`.

## Chạy trên Vercel không cần Python backend

```bash
python -m webapp.export_static_web
```

Script sẽ xuất bản web **chỉ khi** đã đủ 3 checkpoint Q3, Q4, Q5 và bảng chữ cái Q4. Tạo thư mục `docs/Vercel_Release` chứa `index.html`, `browser-infer.js` và `models-public/*.json`. Upload **nội dung thư mục này** lên Vercel Drop hoặc deploy làm static site.

Các phép toán trình duyệt dùng các Tensor xuất từ chính PyTorch: Q2 MLP trên 200 điểm sin mẫu; Q3 MNIST classifier; Q4 RNN theo đúng 47 ký tự của dataset; Q5 CNN; Q6 Attention với trọng số khởi tạo cố định seed 188 (không được mô tả là đã train). Q1 chạy vòng cập nhật Perceptron minh họa bằng JavaScript.

Nếu chỉ deploy `docs/index.html` và `docs/browser-infer.js` mà chưa xuất `models-public`, Q3/Q4/CNN Q5 **không thể suy luận thật**. Tuyệt đối không hiển thị nhãn dự đoán giả.

**Lưu ý:** Trọng số JSON trở thành công khai khi đăng website. Không upload dữ liệu huấn luyện `.npz` lên web/ZIP nộp môn học. Thử độ chính xác với autograder Berkeley trước khi khẳng định điểm bản code mới.
