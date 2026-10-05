# Hướng dẫn tái lập và thông tin bài nộp DeepWeeds — 2A202602954 Phan Trọng Hoàn

> **Vị trí bài nộp (`README.md` mục 5):**
> Thư mục này (`submissions/2A202602954_PhanTrongHoan/`) là gói nộp bài cá nhân chính thức theo quy định tại `README.md` mục 5 của đề bài. File này là `README.md` riêng của bài nộp, mô tả link notebook Kaggle đã chạy, thứ tự thực thi, cấu hình và cách chạy lại toàn bộ quy trình.

---

## 1. Link Notebook Kaggle đã chạy thực tế

- **Link phiên chạy chính thức (Version 9):**
  [https://www.kaggle.com/code/trhphan/track4-day2?scriptVersionId=355337320](https://www.kaggle.com/code/trhphan/track4-day2?scriptVersionId=355337320)
  *Notebook Title:* `track4-day2` | *Script Version:* 9 | *Author:* `trhphan`
- **Mã nguồn notebook tương đương trong repo:** [`code/lab_day2.ipynb`](code/lab_day2.ipynb)

### Quan hệ giữa các phiên (Versions) trên Kaggle
Version 9 là phiên chạy thực thi khóa cấu hình và suy luận tập test cuối cùng cho cấu hình chung kết (`F01`) và mốc so sánh (`T00`) trên 3 seed. **Version 9 không tự chạy lại toàn bộ các thí nghiệm ở các phiên trước**, mà kế thừa kết quả từ chuỗi thực nghiệm có kiểm soát:
- **Version 1:** Thiết lập baseline ban đầu (`ResNet-50` seed 0), kiểm tra chia dữ liệu fold 0 (10.501 train / 3.501 validation / 3.507 test; giao rỗng; hợp đủ 17.509 ảnh), kiểm tra loss ban đầu $\approx \ln(9) \approx 2{,}197$ và overfit batch nhỏ.
- **Version 2:** Sàng lọc 5 backbone (`B01` ResNet-50, `B02` ResNeXt-50, `B03` ConvNeXt-Tiny, `B04` DeiT-Small, `B05` EfficientNet-B0) trên cùng công thức nền validation.
- **Version 3:** Sàng lọc 11 công thức huấn luyện (`T00`–`T10`) trên `ConvNeXt-Tiny` (khởi tạo scratch/frozen/finetune, augmentation cơ bản/màu/RandAug/Mixup/CutMix, loss CE/Label Smoothing/Focal/Weighted CE, EMA).
- **Version 4:** Thí nghiệm kết hợp công thức `T99` (CutMix + Focal loss).
- **Version 6:** Thử nghiệm 9 phương pháp suy luận (`I00`–`I08`) trên checkpoint tốt nhất `T06` (1-view 224, horizontal flip probability/logit, five-crop, dò độ phân giải 256/288/320, temperature scaling, BN fusion).
- **Version 7:** Benchmark đo độ trễ chuẩn hóa trên GPU Tesla T4 (FP32, AMP, FP16 ở batch 1 và batch 32 cho `I04_256`).
- **Version 8:** Khóa cấu hình (`locked_config.json`), huấn luyện lại và kiểm chứng validation trên 3 seed cho `F01` và mốc `T00`.
- **Version 9:** Chạy inference độc lập đúng 1 lượt trên toàn bộ 3.507 ảnh tập test fold 0 cho `F01` và `T00` (seed 0, 1, 2) với nhiệt độ khóa $T = 1{,}0$, xuất `test_attempts.zip` và các file `predictions/`.
- **Version 10:** Xuất bảng tổng hợp `results.xlsx` và kiểm kê artifact.

---

## 2. Môi trường và phiên bản thư viện đã xác minh

Các thông số dưới đây được trích xuất và xác minh trực tiếp từ log thực thi và artifact JSON chính thức của các phiên chạy:
- **Phần cứng:** GPU NVIDIA Tesla T4 (16 GB VRAM) trên nền tảng Kaggle Notebooks.
- **Hệ điều hành / Python:** Linux, Python `3.13.15`.
- **PyTorch:** `2.11.0+cu128` (CUDA 12.8 runtime).
- **torchvision:** `0.26.0+cu128`.
- **timm:** `1.0.29`.
- **fvcore:** `0.1.5.post20221221` (hỗ trợ tính toán tham số và GMAC).
- **Mixed Precision:** Bật AMP (`torch.cuda.amp.autocast`).

---

## 3. Thứ tự chạy Notebook và các cờ điều khiển quan trọng

Mã nguồn tại [`code/lab_day2.ipynb`](code/lab_day2.ipynb) được tổ chức theo cơ chế an toàn: **tất cả cờ chạy đều mặc định là `False`** để ngăn chặn việc vô tình ghi đè hoặc chạy lại các bước tốn kém tài nguyên. Thứ tự chạy và các cờ chính:

1. **Chuẩn bị môi trường & Dữ liệu:**
   - `USE_GOOGLE_DRIVE = False`: Bật nếu chạy trên Google Colab.
   - `RUN_REPO_SETUP = False`: Bật nếu cần clone/pull repo.
   - `RUN_PREPARE_DATA = False`: Giải nén `images.zip` và chuẩn bị thư mục `data/` (kiểm tra MD5 `b7b30f96d466fba86016aa5a26606e0f`).
2. **Kiểm tra sơ bộ (EDA & Sanity Checks):**
   - `RUN_EDA = False`: Phân tích phân bố 9 lớp và kiểm tra ranh giới dữ liệu (S1–S4).
   - `RUN_SANITY_CHECKS = False`: Kiểm tra loss ban đầu $\approx 2{,}118$ và overfit 1 batch nhỏ (<80 bước đưa CE về $\approx 5{,}96 \times 10^{-6}$).
3. **Thực nghiệm Sàng lọc trên Validation (Bước 1 – Bước 3):**
   - `RUN_BASELINE = False`: Chạy baseline mốc ban đầu.
   - `RUN_BACKBONES = False`: Chạy 5 backbone B01–B05.
   - `RUN_TRAINING_ABLATIONS = False`: Chạy các trục công thức huấn luyện T00–T10 và T99.
   - `RUN_INFERENCE_EXPERIMENTS = False`: Chạy các phương pháp suy luận I00–I08.
   - `RUN_LATENCY = False`: Đo độ trễ p50/p95/p99 với 10 lượt warmup, 50 lần đo, đồng bộ `torch.cuda.synchronize()`.
4. **Khóa cấu hình Chung kết và Suy luận Test (Bước 4):**
   - Cấu hình khóa được khai báo rõ ràng trong notebook:
     ```python
     FINAL_BACKBONE = "convnext_tiny"
     FINAL_TRAINING_CONFIG = {"mix": "cutmix", "mix_alpha": 1.0, "loss": "ce"}
     FINAL_INFERENCE_METHOD = "I04_256"
     FINAL_SEEDS = [0, 1, 2]
     ```
   - Chạy huấn luyện lại 3 seed: `RUN_FINAL_TRAINING = True`.
   - Để mở suy luận test một lần duy nhất, notebook yêu cầu xác nhận 2 cờ an toàn:
     ```python
     CONFIRM_FINAL_TEST = "I_HAVE_LOCKED_THE_CONFIGURATION"
     CONFIRM_NO_TEST_PREDICTIONS = True
     RUN_FINAL_TEST = True
     ```
5. **Đánh giá và Xuất Báo cáo:**
   - `RUN_EVAL_REPORTS = False`: Gọi `eval.py score` và `eval.py grade`.
   - `RUN_EXPORT_RESULTS = False`: Tạo bảng Excel `results.xlsx` đầy đủ 7 sheet.
   - `RUN_ARTIFACT_CHECK = False`: Kiểm tra tính toàn vẹn của các file kết quả.

---

## 4. Cấu hình Chung kết F01

Cấu hình tốt nhất được lựa chọn hoàn toàn dựa trên tập validation:
- **Backbone:** `convnext_tiny` (timm).
- **Khởi tạo:** Trọng số tiền huấn luyện ImageNet (`init=finetune`), thay head phân loại 9 lớp, tinh chỉnh toàn bộ mô hình.
- **Công thức huấn luyện:**
  - Augmentation: CutMix với tham số $\alpha = 1{,}0$ (`mix="cutmix"`).
  - Hàm loss: Cross-Entropy tiêu chuẩn (`loss="ce"`).
  - Optimizer: AdamW, Learning Rate backbone $1 \times 10^{-4}$, Learning Rate head $1 \times 10^{-3}$, weight decay $0{,}05$ (không áp dụng cho bias và norm layers).
  - Lịch học: Warmup 1 epoch, sau đó giảm theo Cosine Annealing về 0.
  - Batch size: 64.
  - Số epoch: 12 epochs.
  - Mixed Precision: AMP (`torch.cuda.amp.autocast`).
  - Lựa chọn checkpoint: Epoch có macro-F1 validation cao nhất (epoch 10 hoặc 11 tùy seed).
- **Phương pháp suy luận:**
  - Single-view kích thước $256 \times 256$ (`I04_256`), không dùng TTA.
  - Hiệu chuẩn nhiệt độ: $T = 1{,}0$ (khớp trên validation xác nhận $T=1{,}0$, không áp dụng scaling làm thay đổi xác suất test).
- **Đánh giá đa seed:** Chạy độc lập trên 3 hạt giống: `seed = 0`, `seed = 1`, `seed = 2`.

---

## 5. Đường dẫn File Dự đoán (`predictions/`)

Repo bao gồm đầy đủ 18 file dự đoán tại thư mục `predictions/` (và bản sao trong `submissions/2A202602954_PhanTrongHoan/predictions/`), mỗi file đều có đúng cấu trúc: `Filename, y_true, y_pred, p0, p1, p2, p3, p4, p5, p6, p7, p8`:

- **Dự đoán Test chung kết F01 (3 seed, mỗi file đúng 3.507 dòng dữ liệu):**
  - `predictions/F01_seed0_test.csv`
  - `predictions/F01_seed1_test.csv`
  - `predictions/F01_seed2_test.csv`
- **Dự đoán Test mốc so sánh T00 (ResNet-50 + I00, 3 seed, mỗi file đúng 3.507 dòng dữ liệu):**
  - `predictions/T00_seed0_test.csv`
  - `predictions/T00_seed1_test.csv`
  - `predictions/T00_seed2_test.csv`
- **Dự đoán Validation của F01 và T00 (hỗ trợ chấm I4b, mỗi file 3.501 dòng dữ liệu):**
  - `predictions/F01_seed0_val.csv`, `predictions/F01_seed1_val.csv`, `predictions/F01_seed2_val.csv`
  - `predictions/T00_seed0_val.csv`, `predictions/T00_seed1_val.csv`, `predictions/T00_seed2_val.csv`
- **Dự đoán Test chưa hiệu chuẩn nhiệt độ (uncalibrated, hỗ trợ chấm I4a):**
  - `predictions/F01uncal_seed0_test.csv`, `predictions/F01uncal_seed1_test.csv`, `predictions/F01uncal_seed2_test.csv`
  - `predictions/T00uncal_seed0_test.csv`, `predictions/T00uncal_seed1_test.csv`, `predictions/T00uncal_seed2_test.csv`
  *(Do $T = 1{,}0$, các file uncal trùng khớp byte với các file test đã hiệu chuẩn).*

---

## 6. Hướng dẫn chạy Đánh giá với `eval.py`

Khi người chấm bổ sung hai file nhãn chính thức từ GitHub của tác giả DeepWeeds vào `data/labels/`:
1. `data/labels/labels.csv`
2. `data/labels/test_subset0.csv`

Các lệnh sau có thể được thực thi trực tiếp từ thư mục gốc của repository (hoặc trỏ đường dẫn file tương ứng):

### 6.1. Tính chỉ số chi tiết cho từng cấu hình (`eval.py score`)
```bash
# 1. Tính chỉ số cho cấu hình chung kết F01 (trung bình 3 seed)
python eval.py score --pred "predictions/F01_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv \
    --tag F01 --out eval_out

# 2. Tính chỉ số cho mốc so sánh T00 (trung bình 3 seed)
python eval.py score --pred "predictions/T00_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv --labels data/labels/labels.csv \
    --tag T00 --out eval_out
```

### 6.2. Tự chấm điểm Phần I của RUBRIC (`eval.py grade`)
```bash
python eval.py grade \
    --final "predictions/F01_seed*_test.csv" \
    --baseline "predictions/T00_seed*_test.csv" \
    --test-csv data/labels/test_subset0.csv \
    --labels data/labels/labels.csv \
    --uncal "predictions/F01uncal_seed*_test.csv" \
    --final-val "predictions/F01_seed*_val.csv" \
    --latency-p95-ms 10.407006
```

---

## 7. Trạng thái Minh chứng và Sản phẩm Còn thiếu

Theo quy định của `RUBRIC.md` và nguyên tắc trung thực học thuật:
- **Đã hoàn thiện:**
  - `results.xlsx`: Đủ 7 sheet dữ liệu (`Backbones`, `Training`, `Inference`, `Final`, `PerClass`, `Latency`, `Summary`).
  - `report.md`: Báo cáo chi tiết, trung thực, không nhầm lẫn val/test, không tuyên bố sai về hiệu chuẩn nhiệt độ khi $T=1{,}0$.
  - `curves/`: 6 ảnh biểu đồ huấn luyện thật cho chung kết F01 (seed 0, 1, 2) và mốc T00 (seed 0, 1, 2).
  - `report_assets/`: 2 ảnh ma trận nhầm lẫn (`confusion_matrix_F01.png`, `confusion_matrix_T00.png`) và các bảng CSV metric.
  - `predictions/`: Đủ 6 file test (3.507 dòng) và 12 file validation/uncal hỗ trợ.
- **Các sản phẩm còn thiếu và cách khắc phục khi cần:**
  - **17 ảnh đường cong huấn luyện B/T:** Gồm B01–B05 (5 ảnh từ Version 2), T00–T10 (11 ảnh từ Version 3), T99 (1 ảnh từ Version 4). Do bộ tải trước chỉ tải log/JSON mà không tải PNG, repository không tạo ảnh giả. Có thể tải bổ sung trực tiếp từ thư mục output của các Version tương ứng trên Kaggle.
  - **Artifact EDA và ảnh mẫu:** Không có ảnh JPEG gốc của dataset trong repo; không tạo "ảnh lỗi" giả từ file CSV.
  - **Hai file nhãn gốc:** `data/labels/labels.csv` và `data/labels/test_subset0.csv` chưa được đính kèm trong git repo (để tránh commit dữ liệu lớn). Sau khi tải từ GitHub của tác giả, có thể chạy ngay lệnh `eval.py` ở mục 6.
