# Q6 — Scaled dot-product attention có mặt nạ nhân quả

CS188 Project 5 · Nhóm 9 · `Feature-Q6`

`Attention` nhận tensor `B × T × C`, trong đó `B` là số mẫu, `T` là độ dài chuỗi và `C` là số đặc trưng. Đầu ra có cùng kích thước với đầu vào.

## Truyền xuôi

1. Chiếu đầu vào qua ba lớp tuyến tính riêng để tạo `Q`, `K` và `V`.
2. Tính điểm tương tác `QKᵀ / sqrt(C)`. Việc chia theo căn bậc hai của kích thước đặc trưng giúp giữ độ lớn điểm số ổn định khi `C` tăng.
3. Dùng mặt nạ tam giác dưới. Tại vị trí `t`, điểm của các vị trí sau `t` được đặt thành `-∞`.
4. Tính Softmax theo chiều khóa rồi nhân các trọng số attention với `V` để thu được đầu ra.

| Tệp | Phần cần đọc |
|---|---|
| [models.py](models.py) | `Attention`, ba phép chiếu và mặt nạ |
| [autograder.py](autograder.py) | `check_attention()` |

## Kiểm thử

```bash
python autograder.py -q q6 --no-graphics
```

Lượt chấm riêng Q6 trên CPU với seed 0 đạt **2/2**. Autograder đối chiếu đầu ra với phép tính tham chiếu ở các độ dài khối 2, 4 và 16.

[Cài đặt và hướng dẫn chung](https://github.com/Duck-White/CS188-Project5-Group9/blob/main/README.md)
