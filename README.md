# Q2 — Hồi quy phi tuyến xấp xỉ sin(x)

CS188 Project 5 · Nhóm 9 · `Feature-Q2`

Mô hình học quan hệ từ `x` đến `sin(x)` trên khoảng `[-2π, 2π]`. Mỗi batch có đầu vào và đầu ra kích thước `B × 1`.

## Mô hình và huấn luyện

- MLP `1 → 128 → 128 → 1` dùng ReLU sau hai lớp ẩn.
- `regression_loss()` tính bình phương sai số trung bình (MSE) giữa dự đoán và giá trị `sin(x)`.
- `train_regression()` dùng 200 điểm huấn luyện, batch 50 và Adam với tốc độ học 0,01. Vòng lặp chạy tối đa 300 epoch và dừng sớm khi MSE trung bình của các batch trong một epoch nhỏ hơn 0,005.

| Tệp | Phần cần đọc |
|---|---|
| [models.py](models.py) | `RegressionModel` |
| [losses.py](losses.py) | `regression_loss()` |
| [train.py](train.py) | `train_regression()` |

## Kiểm thử

```bash
python autograder.py -q q2 --no-graphics
```

Lượt chấm riêng Q2 trên CPU với seed 0 đạt **5/5**, MSE train **0,005437**. Ngưỡng MSE train để lấy đủ điểm là 0,02.

[Cài đặt và hướng dẫn chung](https://github.com/Duck-White/CS188-Project5-Group9/blob/main/README.md)
