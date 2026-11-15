# Q3 — Nhận dạng chữ số MNIST bằng MLP

CS188 Project 5 · Nhóm 9 · `Feature-Q3`

Mỗi ảnh MNIST 28 × 28 được trải thành 784 giá trị. Mô hình gán ảnh vào một trong 10 lớp từ 0 đến 9.

## Dữ liệu, mô hình và huấn luyện

- `DigitClassificationDataset` dùng 60.000 ảnh train, 5.000 ảnh dev và 5.000 ảnh test từ `data/mnist.npz`.
- MLP `784 → 256 → 128 → 10` dùng ReLU ở hai lớp ẩn. Mô hình trả 10 logits. Lớp có logit lớn nhất là chữ số dự đoán.
- Vòng lặp dùng Cross-Entropy, Adam với tốc độ học 0,001 và batch 100. Mô hình được huấn luyện tối đa 20 epoch. Từ epoch 5, vòng lặp có thể dừng nếu accuracy dev tốt nhất đạt 98%.
- Lưu và khôi phục trọng số tại epoch có accuracy dev cao nhất để chấm trên test.

| Tệp | Phần cần đọc |
|---|---|
| [models.py](models.py) | `DigitClassificationModel` |
| [losses.py](losses.py) | `digitclassifier_loss()` |
| [train.py](train.py) | `train_digitclassifier()` |

## Kiểm thử

```bash
python autograder.py -q q3 --no-graphics
```

Lượt chấm riêng Q3 trên CPU với seed 0 đạt **5/5**. Accuracy dev tốt nhất **98,00%** tại epoch 5. Accuracy test **97,46%** (4.873/5.000 ảnh).

[Cài đặt và hướng dẫn chung](https://github.com/Duck-White/CS188-Project5-Group9/blob/main/README.md)
