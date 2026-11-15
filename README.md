# Q5 — Tích chập 2D để phân loại chữ số

CS188 Project 5 · Nhóm 9 · `Feature-Q5`

`Convolve()` trượt kernel 3 × 3 trên ảnh 28 × 28 với bước 1. Mỗi ô đầu ra là tổng các tích giữa cửa sổ ảnh và kernel. Bản đồ đặc trưng thu được có kích thước 26 × 26.

## Mô hình và huấn luyện

- `DigitConvolutionalModel` dùng một kernel có tham số học được. Sau tích chập, mô hình trải phẳng 676 giá trị rồi đưa qua MLP `676 → 200 → 10`, với ReLU ở lớp ẩn.
- `digitconvolution_Loss()` dùng Cross-Entropy. `Train_DigitConvolution()` huấn luyện 10 epoch, batch 32, Adam với tốc độ học 0,001.
- Bộ nạp Q5 lấy 200 ảnh đầu của tập train MNIST. Trong 100 ảnh đầu của nhóm này, các chỉ số chẵn tạo tập dev 50 ảnh và các chỉ số lẻ tạo tập test 50 ảnh. Hai tập đo này cùng nằm trong 200 ảnh huấn luyện.

| Tệp | Phần cần đọc |
|---|---|
| [models.py](models.py) | `Convolve()`, `DigitConvolutionalModel` |
| [losses.py](losses.py) | `digitconvolution_Loss()` |
| [train.py](train.py) | `Train_DigitConvolution()` |
| [backend.py](backend.py) | `DigitClassificationDataset2` |

## Kiểm thử

```bash
python autograder.py -q q5 --no-graphics
```

Lượt chấm riêng Q5 trên CPU với seed 0 đạt **3/3** và đúng **50/50 ảnh** trong tập test mà Autograder Q5 sử dụng. Khi gọi hàm huấn luyện, Autograder dùng đúng cách viết `Train_DigitConvolution`.

[Cài đặt và hướng dẫn chung](https://github.com/Duck-White/CS188-Project5-Group9/blob/main/README.md)
