# Bài nộp Lab Day 2 DeepWeeds — 2A202602954 Phan Trọng Hoàn

## Mở notebook và chạy lại

- [Mở notebook trên Google Colab](https://colab.research.google.com/github/naoh-pt/K4-Track4-Day2-PhanTrongHoan-2A202602954/blob/main/submissions/2A202602954_PhanTrongHoan/code/lab_day2.ipynb).
- [Mã notebook trong bài nộp](code/lab_day2.ipynb). Trên Kaggle, import file này vào một Kaggle Notebook, attach DeepWeeds Dataset và bật Internet nếu cần clone repo/cài dependency.
- Repo nguồn: <https://github.com/naoh-pt/K4-Track4-Day2-PhanTrongHoan-2A202602954>.

Notebook có cờ mặc định `False` cho các thao tác cài gói, chuẩn bị dữ liệu, huấn luyện, suy luận và đo độ trễ. Hãy đọc các cell từ đầu, chỉ bật cờ của bước đang chạy. Notebook trong Git được giữ sạch output.

## Dữ liệu và thứ tự chạy

1. Dùng **fold 0 nguyên bản** của DeepWeeds. Với Colab, đặt `images.zip` và `labels/` trong `/content/drive/MyDrive/DeepWeeds`, bật `USE_GOOGLE_DRIVE`, cấu hình đường dẫn và `RUN_PREPARE_DATA`. Notebook kiểm tra MD5 ZIP `b7b30f96d466fba86016aa5a26606e0f` và tổng 17.509 ảnh. Với Kaggle, thay `<dien-dataset-slug>` bằng Dataset đã attach; output nằm dưới `/kaggle/working`.
2. Dùng runtime có PyTorch/torchvision sẵn. Bật `INSTALL_DEPS` khi cần cài `timm`, `fvcore`, `openpyxl`, `pytest`; notebook không cài lại PyTorch. Bật `RUN_REPO_SETUP` nếu cần clone/pull repo.
3. Chạy EDA và sanity check, rồi lần lượt chạy baseline, năm backbone, các recipe và phương pháp suy luận. Mọi lựa chọn dựa vào **validation**. Các cấu hình huấn luyện đi qua cùng `train.run(Config(...))`.
4. Điền `FINAL_BACKBONE`, `FINAL_TRAINING_CONFIG`, `FINAL_INFERENCE_METHOD` theo validation; chạy F01 và baseline T00 cùng seed `0, 1, 2`. Checkpoint được chọn bằng macro-F1 validation. Chỉ mở test sau khi khóa cấu hình và xác nhận bằng cờ trong notebook; mỗi seed chỉ có một lượt suy luận test. Không dùng test để chọn lại.
5. Chạy `eval.py score` cho F01/T00 và `eval.py grade` trên prediction thật, rồi xuất workbook và lưu curves, prediction, JSON và log vào Drive hoặc Kaggle Output. Đưa artifact nhỏ vào các thư mục tương ứng của bài nộp; không đưa dataset hay checkpoint lớn vào Git.

Một Filename chính thức (`20170714-110407-3.jpg`) có Label khác giữa `labels.csv` và `train_subset0.csv`. `code/dataset.py` giữ Label của subset làm target, suy ra Species theo lớp và phát cảnh báo; không sửa CSV.

## Môi trường và kết quả hiện được bàn giao

Theo thông tin phiên chạy được cung cấp: Tesla T4; Python 3.13.15; PyTorch 2.11.0+cu128; torchvision 0.26.0+cu128; AMP bật. Seed chung kết và mốc: `0, 1, 2`. [CẦN BỔ SUNG: version `timm`, `fvcore`, tag trọng số ImageNet và đường dẫn/ID phiên Kaggle hoặc bản Colab đã chạy.]

`results.xlsx` hiện chứa **sáu dòng validation final F01/T00** trong sheet `Inference` và hai dòng tổng hợp trong `Summary`; các sheet còn lại phần lớn chỉ có header. `report.md` dùng thêm số liệu do người thực hiện cung cấp trong yêu cầu viết báo cáo, có ghi rõ phần chưa thể đối chiếu. Workbook không thay thế prediction CSV và log gốc.

`curves/` và `predictions/` hiện chỉ có `.gitkeep` để giữ cấu trúc trong Git. [CẦN BỔ SUNG: curves theo từng exp_id; `F01_seed0/1/2_test.csv`, `T00_seed0/1/2_test.csv` và các file validation/uncal nếu có; output `eval.py score`/`grade`; JSON/config/history từ runtime.]

Các module Python và notebook nằm trong `code/`. Bộ test tự viết và `eval.py` gốc vẫn ở repo nguồn; chạy từ root repo khi đã có dependency. Trước khi nộp cuối, đối chiếu các bảng trong báo cáo và workbook với prediction bằng `eval.py`, bảo đảm mỗi file test có 3.507 ảnh và đủ cột `Filename, y_true, y_pred, p0…p8`.
