# Q1 — Phân loại nhị phân bằng Perceptron

CS188 Project 5 · Nhóm 9 · `Feature-Q1`

Mục tiêu của Q1 là gán một điểm vào lớp `+1` hoặc `-1`. Dữ liệu mặc định có hai tọa độ và một thành phần hằng cho bias.

## Triển khai

1. Khởi tạo vector trọng số `w` bằng 1. `forward()` lấy tích vô hướng giữa đầu vào và trọng số. `get_prediction()` trả `+1` khi điểm số từ 0 trở lên và `-1` khi điểm số âm.
2. `train_perceptron()` duyệt từng mẫu. Với mẫu có nhãn `y` bị phân loại sai, cập nhật `w ← w + yx`.
3. Dừng sau một lượt qua tập huấn luyện mà mọi mẫu đều được phân loại đúng.

| Tệp | Phần cần đọc |
|---|---|
| [models.py](models.py) | `PerceptronModel`, `get_weights()` và `get_prediction()` |
| [train.py](train.py) | `train_perceptron()` |
| [backend.py](backend.py) | `PerceptronDataset` |

## Kiểm thử

```bash
python autograder.py -q q1 --no-graphics
```

Lệnh trên tại `Feature-Q1` hiện dừng khi nhập `losses.py` vì lỗi `SyntaxError` ở dòng 86. Bản Q1 trên `main` đã chấm đạt **5/5**. [Bản mã Q1 trên main đã kiểm thử](https://github.com/Duck-White/CS188-Project5-Group9/commit/915baf949497da79825f44c6838c5a6c12244ce4)

[Cài đặt và hướng dẫn chung](https://github.com/Duck-White/CS188-Project5-Group9/blob/main/README.md)
