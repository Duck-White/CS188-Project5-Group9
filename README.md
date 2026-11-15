# Q4 — Nhận diện ngôn ngữ bằng RNN

CS188 Project 5 · Nhóm 9 · `Feature-Q4`

Q4 dự đoán ngôn ngữ từ cách viết của một từ. Năm nhãn là English, Spanish, Finnish, Dutch và Polish.

## Dữ liệu, mô hình và huấn luyện

- Mỗi ký tự thuộc bảng 47 ký tự được mã hóa one-hot. Dữ liệu lấy từ `data/lang_id.npz`. Vòng lặp đổi thứ tự tensor từ `B × T × 47` sang `T × B × 47` để xử lý từng bước thời gian.
- RNN dùng trạng thái ẩn 256 chiều. Ở mỗi bước, cộng phép chiếu ký tự với phép chiếu trạng thái trước rồi áp dụng ReLU. Hàng ký tự đệm toàn 0 giữ nguyên trạng thái.
- Trạng thái cuối đi qua lớp 128 nút với ReLU và lớp 5 logits. Mô hình chọn ngôn ngữ có logit lớn nhất.
- Huấn luyện bằng Cross-Entropy và Adam, batch 128. Tốc độ học ban đầu 0,001, giảm còn 0,0003 từ epoch 21 nếu quá trình còn tiếp tục. Từng phần tử gradient được giới hạn trong `[-1, 1]`.
- Lưu mô hình có accuracy dev cao nhất. Tối đa 40 epoch và có thể dừng từ epoch 10 khi accuracy dev tốt nhất đạt 85%.

| Tệp | Phần cần đọc |
|---|---|
| [models.py](models.py) | `LanguageIDModel` |
| [losses.py](losses.py) | `languageid_loss()` |
| [train.py](train.py) | `train_languageid()` |

## Kiểm thử

```bash
python autograder.py -q q4 --no-graphics
```

Lượt chấm riêng Q4 trên CPU với seed 0 đạt **5/5**. Accuracy dev tốt nhất **90,80%** tại epoch 8. Accuracy test **90,20%** (451/500 từ).

[Cài đặt và hướng dẫn chung](https://github.com/Duck-White/CS188-Project5-Group9/blob/main/README.md)
