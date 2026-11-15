# CS188 Project 5 — Machine Learning

Nhóm 9 · INT3401_11

`main` tổng hợp sáu bài của Project 5. Mỗi nhánh `Feature-Qx` có README riêng giải thích mô hình, các hàm cần đọc và cách chấm câu đó. [Đề bài CS188 Project 5](https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/)

| Câu | Bài toán | Mô hình hoặc hàm chính | Hướng dẫn |
|---|---|---|---|
| Q1 | Phân loại nhị phân | `PerceptronModel`, `train_perceptron` | [Feature-Q1](https://github.com/Duck-White/CS188-Project5-Group9/blob/Feature-Q1/README.md) |
| Q2 | Xấp xỉ hàm sin(x) | `RegressionModel`, `regression_loss` | [Feature-Q2](https://github.com/Duck-White/CS188-Project5-Group9/blob/Feature-Q2/README.md) |
| Q3 | Nhận dạng chữ số MNIST | `DigitClassificationModel`, mạng MLP | [Feature-Q3](https://github.com/Duck-White/CS188-Project5-Group9/blob/Feature-Q3/README.md) |
| Q4 | Nhận diện ngôn ngữ của một từ | `LanguageIDModel`, mạng RNN | [Feature-Q4](https://github.com/Duck-White/CS188-Project5-Group9/blob/Feature-Q4/README.md) |
| Q5 | Phân loại ảnh với tích chập 2D | `Convolve`, `DigitConvolutionalModel` | [Feature-Q5](https://github.com/Duck-White/CS188-Project5-Group9/blob/Feature-Q5/README.md) |
| Q6 | Tự chú ý có mặt nạ nhân quả | `Attention` | [Feature-Q6](https://github.com/Duck-White/CS188-Project5-Group9/blob/Feature-Q6/README.md) |

## Cài đặt và chạy

Clone repo rồi mở terminal trong thư mục dự án. Cài PyTorch, NumPy và Matplotlib trong môi trường Python đang dùng.

```bash
git clone https://github.com/Duck-White/CS188-Project5-Group9.git
cd CS188-Project5-Group9
python -m pip install torch numpy matplotlib
python autograder.py --no-graphics
```

Để chấm từng câu, thêm `-q` và mã câu. Lệnh sau chạy riêng Q3.

```bash
python autograder.py -q q3 --no-graphics
```

Dữ liệu `data/mnist.npz` dùng cho Q3 và Q5. Q4 dùng `data/lang_id.npz`. Các tệp này đã có trong repo.

## Kết quả kiểm thử trên main

Lượt chấm cả sáu câu dùng CPU, seed 0 và Python 3.12.14. [Bản mã trên main được kiểm thử](https://github.com/Duck-White/CS188-Project5-Group9/commit/915baf949497da79825f44c6838c5a6c12244ce4)

| Câu | Điểm | Kết quả |
|---|---:|---|
| Q1 | 5/5 | Phân loại đúng 100% mẫu huấn luyện |
| Q2 | 5/5 | MSE train 0,003455 |
| Q3 | 5/5 | Dev tốt nhất 98,00%, test 97,82% |
| Q4 | 5/5 | Dev tốt nhất 91,00%, test 90,00% |
| Q5 | 1/3 | Qua kiểm tra tích chập. Autograder gọi `Train_DigitConvolution`, còn hàm trong `main` có tên `train_digitconvolution` |
| Q6 | 2/2 | Qua kiểm tra đầu ra attention |
| **Tổng** | **23/25** | |

Bản thử cục bộ thêm dòng `Train_DigitConvolution = train_digitconvolution` trong `train.py` đạt **25/25**.

## Tệp nguồn

| Tệp | Vai trò |
|---|---|
| [models.py](models.py) | Kiến trúc, hàm truyền xuôi và tích chập |
| [losses.py](losses.py) | MSE và Cross-Entropy |
| [train.py](train.py) | Vòng lặp huấn luyện Q1–Q5 |
| [backend.py](backend.py) | Nạp, chia và mã hóa dữ liệu |
| [autograder.py](autograder.py) | Kiểm tra điều kiện riêng của từng câu |
