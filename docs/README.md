# ML LAB — Website Q1–Q6

Website trong `docs/index.html` là giao diện tĩnh, hỗ trợ cả màn hình điện thoại và máy tính. Không cần backend hay tài khoản để mở trực tiếp file HTML.

Nội dung:
- 6 mô-đun Q1–Q6, kết quả từ nhật ký cũ và liên kết mã nguồn.
- Mô phỏng tương tác Q1 (Perceptron), Q2 (Regression), Q6 (Causal Attention).
- Ghi rõ demo là mô phỏng trong trình duyệt; không suy luận từ checkpoint PyTorch.

## Chạy cục bộ

```bash
python -m http.server 8000 --directory docs
```

Truy cập `http://localhost:8000`.

## Đưa lên GitHub Pages

Sau khi review/merge nhánh này, kiểm tra hỗ trợ GitHub Pages cho quyền riêng tư của repo. Trong Settings → Pages, chọn Deploy from a branch → `main` → `/docs`. Website chỉ hoạt động công khai khi Pages được kích hoạt và triển khai thành công. Cần cân nhắc công khai mã nguồn/báo cáo/dữ liệu trước khi publish.

## Kiểm thử

Bản trước tối ưu Q5 có hai nhật ký đạt 25/25. Bản tối ưu đã chấm lại Q1 (5/5), Q2 (5/5), Q6 (2/2) và vượt qua smoke test Q1–Q6. Chưa chấm lại Q3–Q5 trên dữ liệu Berkeley đầy đủ. Đề bài: https://inst.eecs.berkeley.edu/~cs188/archive/fa25/projects/proj5/
