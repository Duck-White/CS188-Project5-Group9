# CS188 Project 5 Machine Learning

Nhóm 9 · INT3401_11 · Trí tuệ nhân tạo. Đề tài sử dụng [CS188 Fall 2025 của UC Berkeley](https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/).

| Câu | Bài toán | Phụ trách |
| --- | --- | --- |
| Q1 | Phân loại nhị phân bằng Perceptron | Cường |
| Q2 | Xấp xỉ hàm sin bằng mạng nơ-ron | Dũng |
| Q3 | Phân loại chữ số MNIST bằng MLP | Đạt |
| Q4 | Nhận diện ngôn ngữ của một từ bằng RNN | Đạt |
| Q5 | Phân loại chữ số bằng tích chập 2D | Quân |
| Q6 | Scaled dot-product attention với causal mask | Quân |

## Cài đặt và chạy

Mở terminal tại thư mục có `autograder.py`. Bản kiểm thử dùng Python 3.12.

```bash
python -m venv .venv
```

Kích hoạt môi trường bằng `.venv\Scripts\activate.bat` trong CMD trên Windows,
`.venv\Scripts\Activate.ps1` trong PowerShell hoặc `source .venv/bin/activate`
trên macOS/Linux. Sau đó chạy:

```bash
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python autograder.py --no-graphics
```

Trên macOS, dùng `python -m pip install torch==2.15.0` để cài PyTorch từ PyPI.
Chạy riêng một câu, ví dụ Q3, bằng `python autograder.py -q q3 --no-graphics`.

Khi dùng mã nguồn từ gói nộp, giải nén thư mục `data` của
[bộ mã gốc Fall 2025](https://inst.eecs.berkeley.edu/~cs188/fa25/assets/projects/machinelearning.zip)
vào cạnh `autograder.py`. Cần đủ `data/mnist.npz` và `data/lang_id.npz`.
Nếu dùng `git clone` của nhóm, hai tệp này đã có sẵn.

## Tái lập kết quả

Lệnh sau đặt seed 0 và một luồng CPU, rồi chạy bộ chấm gốc.

```bash
python -c "import random,runpy,numpy,torch; random.seed(0); numpy.random.seed(0); torch.manual_seed(0); torch.set_num_threads(1); runpy.run_path('autograder.py', run_name='__main__')" --no-graphics
python verify_project.py --seed 0
```

| Câu | Điểm | Chỉ số của lần chạy tích hợp seed 0 |
| --- | ---: | --- |
| Q1 | 5/5 | Phân loại đúng 100% tập huấn luyện |
| Q2 | 5/5 | MSE trên dữ liệu bộ chấm: 0,003455 |
| Q3 | 5/5 | Dev tốt nhất 98,00%, test 97,82% |
| Q4 | 5/5 | Dev tốt nhất 91,00%, test 90,00% |
| Q5 | 3/3 | 50/50 ảnh trong bộ chấm |
| Q6 | 2/2 | Mọi ca kiểm thử attention của bộ chấm đều đạt |
| **Tổng** | **25/25** | **CPU, PyTorch 2.15.0+cpu, NumPy 2.3.5** |

Q5 của bộ mã gốc huấn luyện trên 200 ảnh đầu của tập train MNIST. Cả 50 ảnh
bộ chấm dùng để tính accuracy đều thuộc 200 ảnh đó. Lần chạy bổ sung với
`verify_project.py` đạt **69,36%** trên 5.000 ảnh độc lập trong tập test MNIST, với
cùng cấu hình huấn luyện 200 ảnh. Script cũng kiểm tra gradient của phép tích
chập, padding RNN và tính nhân quả của attention.

Các phần triển khai nằm trong `models.py`, `losses.py` và `train.py`. Các tệp
`autograder.py` và `backend.py` giữ đúng nội dung bộ mã Berkeley Fall 2025.
