# Báo cáo Lab Day 2: phân loại DeepWeeds bằng backbone, công thức huấn luyện và suy luận

> **Tình trạng minh chứng.** Các số thực nghiệm trong báo cáo này do người thực hiện cung cấp trong yêu cầu lập báo cáo ngày 05/10/2026. Repo hiện chỉ có mã, notebook, `README.md`, `GUIDE.md` và `RUBRIC.md`; chưa có `results.xlsx`, log chạy, JSON đo suy luận/độ trễ, prediction CSV hay ảnh curves để đối soát độc lập. Những nội dung cần artifact được đánh dấu `[CẦN BỔ SUNG: ...]`. Các công thức, quy tắc chia dữ liệu và cách chấm theo tài liệu của repo; bảng seed được kiểm tra lại bằng độ lệch chuẩn mẫu (`ddof=1`).

## 1. Tóm tắt

DeepWeeds là bài toán phân loại 17.509 ảnh RGB thành chín lớp. Nghiên cứu đã so sánh năm backbone, nhiều công thức huấn luyện và phương pháp suy luận trên fold 0. Cấu hình cuối được chọn bằng validation: ConvNeXt-Tiny khởi tạo bằng trọng số ImageNet, tinh chỉnh toàn bộ, CutMix, suy luận một view ở 256 × 256 (`I04_256`). Theo bảng kết quả được cung cấp, qua ba seed, macro-F1 test đạt **0.977462 ± 0.002598** và top-1 test **0.982226 ± 0.001941**. So với mốc ResNet-50 `T00` dùng một view `I00`, chênh lệch macro-F1 khoảng **0.165805** (16,58 điểm phần trăm) và top-1 khoảng **0.121756** (12,18 điểm phần trăm), lớn hơn rõ rệt độ lệch chuẩn giữa seed. Các con số test cần được đối chiếu lại với prediction CSV trước khi nộp. Cấu hình cuối dùng nhiệt độ **T = 1.0**; báo cáo không quy mức tăng test cho hiệu chuẩn nhiệt độ.

## 2. Dữ liệu và thiết lập

### 2.1. Dữ liệu, split và chỉ số

DeepWeeds gồm 17.509 ảnh RGB kích thước gốc 256 × 256, với tám loài cỏ dại và một lớp `Negative`. Theo thống kê được cung cấp, fold 0 của tác giả có 10.501 ảnh train, 3.501 ảnh validation và 3.507 ảnh test; không tự chia lại hoặc gộp validation vào train. Giao Filename của từng cặp tập bằng 0; hợp ba tập có 17.509 tên. Bảng dưới là số đếm được cung cấp, chưa có CSV trong repo để đếm lại.

| Label | Train | Validation | Test |
|---:|---:|---:|---:|
| 0 · Chinee apple | 675 | 225 | 226 |
| 1 · Lantana | 637 | 213 | 213 |
| 2 · Parkinsonia | 618 | 206 | 207 |
| 3 · Parthenium | 613 | 204 | 205 |
| 4 · Prickly acacia | 637 | 212 | 213 |
| 5 · Rubber vine | 605 | 202 | 202 |
| 6 · Siam weed | 644 | 215 | 215 |
| 7 · Snake weed | 609 | 203 | 204 |
| 8 · Negative | 5.463 | 1.821 | 1.822 |
| **Tổng** | **10.501** | **3.501** | **3.507** |

Lớp `Negative` chiếm khoảng 52% ở cả ba tập. Tỉ số lớp lớn nhất/nhỏ nhất lần lượt là 9.0298, 9.0149 và 9.0198 trên train, validation và test. Mất cân bằng này làm top-1 dễ bị chi phối bởi lớp `Negative`; **macro-F1 trung bình đều trên chín lớp** là chỉ số chọn mô hình chính. Top-1, balanced accuracy, F1 từng lớp và ECE 15 bin là các chỉ số bổ sung theo `README.md` mục 2.2. Tổng 17.509 phù hợp quy mô trong Table 1 được trích ở `README.md`, nhưng phần phân bố theo split vẫn cần xác nhận bằng CSV thật.

Khi cộng ba split theo **Label của subset**, lớp 0 có 1.126 ảnh và lớp 1 có 1.063 ảnh. Table 1 trong `README.md`, theo metadata gốc, ghi lần lượt 1.125 và 1.064. Chênh một ảnh theo hai chiều phù hợp với Filename lệch Label được nêu ở mục 2.2; không được sửa số đếm của subset để ép khớp Table 1.

[CẦN BỔ SUNG: biểu đồ phân bố lớp và lưới 27 ảnh EDA, ít nhất ba ảnh thật cho mỗi lớp; chỉ chèn đường dẫn hình khi artifact tồn tại.]

### 2.2. Kiểm tra pipeline và ranh giới dữ liệu

Theo log được mô tả trong yêu cầu, batch kiểm tra có shape `[8, 3, 224, 224]`, `float32`; dải giá trị sau chuẩn hóa khoảng −2.1179 đến 2.6400. CE ban đầu là 2.118345, gần mốc tham chiếu ln(9) = 2.197225; sau tối đa 80 bước quá khớp một batch nhỏ, CE xuống 5.9604 × 10⁻⁶. Kết quả kiểm thử được cung cấp là 75 pytest và 38 unittest. Cần lưu output kiểm thử và ảnh đầu vào đã giải chuẩn hóa để truy vết các kiểm tra này.

`labels.csv` là metadata; các subset của tác giả quyết định target. Có **một** Filename được báo cáo lệch Label: `20170714-110407-3.jpg` có Label 1/Species `Lantana` trong `labels.csv`, nhưng Label 0 trong `train_subset0.csv`. Pipeline giữ nguyên subset và phát `RuntimeWarning` ghi tên file để truy vết; CSV không được sửa. `Species` trả về được suy ra từ Label của subset. Việc lệch này cần được ghi trong log chạy thật vì nó ảnh hưởng cách đối chiếu metadata.

Train chỉ cập nhật trọng số. Validation chọn backbone, recipe, checkpoint theo macro-F1 cao nhất, phương pháp suy luận và nhiệt độ nếu dùng. Test chỉ được mở sau khi khóa lựa chọn bằng validation; các bảng test ở phần 7 chỉ dùng để báo cáo cuối. Notebook trong `code/lab_day2.ipynb` có cơ chế khóa lựa chọn và cờ xác nhận test; việc thực thi đúng quy trình cần được chứng thực bằng artifact phiên chạy.

### 2.3. Môi trường và công thức nền

Thông tin môi trường được cung cấp: Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128; AMP được bật khi huấn luyện. [CẦN BỔ SUNG: version `timm`, `fvcore`, tag trọng số ImageNet chính xác của từng backbone, link phiên Colab/Kaggle và output `config.json` ghi môi trường.]

Theo cấu hình nền trong notebook và thông số được cung cấp, mỗi backbone khởi tạo ImageNet, thay head chín lớp rồi fine-tune toàn bộ; AdamW có LR backbone 1 × 10⁻⁴, LR head 1 × 10⁻³, weight decay 0.05; warmup một epoch rồi cosine; batch 64, 12 epoch, ảnh train 224, augmentation cơ bản, CE và AMP. Checkpoint tốt nhất được chọn theo macro-F1 validation. Train dùng RandomResizedCrop và lật ngang; đánh giá gốc dùng transform xác định. Bảng backbone dưới đây chỉ công bằng hoàn toàn nếu các `config.json` xác nhận cùng recipe, seed và tag trọng số đã khai báo.

## 3. So sánh backbone trên validation

| Exp | Backbone | Params (M) | GMAC | Macro-F1 val | Top-1 val | Giây/epoch |
|:---|:---|---:|---:|---:|---:|---:|
| B01 | ResNet-50 | 23.526473 | 4.087155 | 0.800512 | 0.856612 | 57.395 |
| B02 | ResNeXt-50 32×4d | 22.998345 | 4.228450 | 0.748919 | 0.793773 | 77.833 |
| B03 | ConvNeXt-Tiny | 27.827049 | 4.454770 | **0.972956** | **0.979720** | 70.036 |
| B04 | DeiT-Small/16-224 | 21.669129 | 4.240838 | 0.948585 | 0.963439 | 48.876 |
| B05 | EfficientNet-B0 | 4.019077 | 0.384546 | 0.752492 | 0.819480 | **37.042** |

Trong lần sàng được cung cấp, ConvNeXt-Tiny có macro-F1 validation cao nhất, hơn ResNet-50 **0.172444**. DeiT-Small đứng thứ hai và mất 48.876 giây/epoch, ít hơn ConvNeXt-Tiny trong phép đo này. EfficientNet-B0 có ít tham số, GMAC và thời gian/epoch thấp nhất, nhưng macro-F1 thấp hơn rõ rệt. GMAC không xếp hạng trùng thời gian huấn luyện: DeiT-Small 4.240838 GMAC chạy nhanh hơn B01 4.087155 GMAC trong bảng này. GMAC cũng không thay thế phép đo latency thực tế. ConvNeXt-Tiny được chọn để đi tiếp vì chất lượng validation nổi bật, đồng thời số p95 được cung cấp cho cấu hình cuối trên T4 đáp ứng ngân sách thời gian thực. Đây là sàng lọc **một seed**; chưa đủ để khẳng định những chênh lệch nhỏ sẽ lặp lại.

[CẦN BỔ SUNG: độ trễ sơ bộ batch 1 cho từng backbone, tag trọng số từ config/log và biểu đồ accuracy–cost lấy từ artifact thật.]

## 4. Công thức huấn luyện trên ConvNeXt-Tiny

`T00` **trong vòng recipe** là ConvNeXt-Tiny với công thức nền, khác với `T00` **mốc chung kết** ở phần 7 là ResNet-50 + `I00`. Cùng mã `T00` ở hai namespace có thể gây nhầm nếu bỏ tên giai đoạn. Các số sau là validation được cung cấp; ánh xạ thí nghiệm thực chạy sang thay đổi cấu hình cần `config.json`/`summary.json` để xác nhận. Notebook có kế hoạch cho T01–T10, nhưng kế hoạch đó không chứng minh cấu hình thực chạy.

| Exp | Thay đổi so với recipe T00 | Best epoch | Macro-F1 val | Top-1 val | Giây/epoch |
|:---|:---|---:|---:|---:|---:|
| T00 | Nền ConvNeXt-Tiny, basic + CE | 12 | 0.972956 | 0.979720 | 69.443 |
| T01 | Chưa xác minh | 11 | 0.851600 | 0.882319 | 42.973 |
| T02 | Chưa xác minh | 12 | 0.305518 | 0.544702 | 68.643 |
| T03 | Chưa xác minh | 12 | 0.968189 | 0.976292 | 78.259 |
| T04 | Chưa xác minh | 10 | 0.973224 | 0.979149 | 68.372 |
| T05 | Chưa xác minh | 10 | 0.965909 | 0.973436 | 69.422 |
| T06 | CutMix | 10 | **0.974513** | **0.980863** | 69.040 |
| T07 | Chưa xác minh | 11 | 0.969879 | 0.976292 | 68.766 |
| T08 | Chưa xác minh | 10 | 0.973130 | 0.980577 | 67.512 |
| T09 | Chưa xác minh | 12 | 0.964355 | 0.972008 | 68.023 |
| T10 | Chưa xác minh | 12 | 0.968571 | 0.975721 | 68.721 |
| T99 | CutMix + focal loss | 10 | 0.971149 | 0.977721 | 81.430 |

T06 cao hơn recipe T00 **0.001557** nếu trừ hai số sáu chữ số trong bảng (yêu cầu nêu khoảng 0.001556 theo số chưa làm tròn). Vì vòng ablation chủ yếu một seed và chưa có std, đây là kết quả tốt nhất *trong lần sàng*, chưa đủ chứng cứ rằng CutMix luôn đem lại lợi thế. T99 thấp hơn T06 **0.003364**, nên kết hợp CutMix với focal loss không cho hiệu ứng cộng dồn trong lần thử này. T02 thấp hơn T00 **0.667438** macro-F1, một thất bại rõ về điểm số; **không gán nguyên nhân** khi chưa có cấu hình thực tế và đường cong. Các kết quả thấp vẫn hữu ích để nhận diện độ nhạy với khởi tạo, loss hoặc augmentation sau khi đối chiếu log.

[CẦN BỔ SUNG: cấu hình chính xác của T01–T05 và T07–T10 từ từng `config.json`/`summary.json`, gồm trục thay đổi, loss, augmentation, sampler, initialization và EMA.]

[CẦN BỔ SUNG: `history.csv` và ảnh curve của B01–B05, T00–T10, T99 để nhận xét cụ thể về tốc độ hội tụ, overfit và dao động theo epoch.]

## 5. Phương pháp suy luận trên validation

Các phương pháp dưới đây được so sánh trên validation của checkpoint được chọn trước đó. `K` là số view; tăng K đòi hỏi thêm forward. `I04_256` là một view ở 256 × 256.

| Mã/phương pháp | K | Macro-F1 val | Top-1 val | ECE val |
|:---|---:|---:|---:|---:|
| I00 · single 224 | 1 | 0.974513 | 0.980863 | 0.008020 |
| I01 · lật ngang, trung bình xác suất | 2 | 0.975133 | 0.981434 | 0.008276 |
| I02 · five-crop | 5 | 0.974889 | 0.981434 | 0.009220 |
| I03 · lật ngang, trung bình logit | 2 | 0.975133 | 0.981434 | 0.008212 |
| I04_256 · single 256 | 1 | **0.976058** | **0.982291** | 0.017985 |
| I04_288 · single 288 | 1 | 0.972385 | 0.978292 | 0.032886 |
| I04_320 · single 320 | 1 | 0.967094 | 0.972579 | 0.042501 |
| I07 · temperature scaling | 1 | 0.974513 | 0.980863 | **0.002411** |
| I08 · Conv–BN fusion | 1 | 0.974513 | 0.980863 | 0.008020 |

`I04_256` đạt macro-F1 validation cao nhất; tăng tiếp lên 288 và 320 làm điểm giảm trong lần đo này. I01/I03 tăng khoảng **0.000620** so với I00 nhưng cần hai forward; five-crop cần năm forward và không vượt I04_256. I04_256 tăng **0.001545** so với I00 theo số hiển thị (khoảng 0.00155), chỉ cần một forward nên hợp lý hơn TTA nhiều view khi có giới hạn độ trễ. Các thay đổi rất nhỏ đều chưa có std qua nhiều seed tại vòng sàng.

I07 khớp trên validation với **T = 0.9070198983**: ECE val từ 0.008020 xuống 0.002411, còn macro-F1/top-1 không đổi do chia logit cho T dương không đổi argmax. I08 bảo toàn các metric val trong bảng. Cấu hình cuối dùng `I04_256`, **không kết hợp** temperature scaling; T final = 1.0. Vì vậy không có bằng chứng I07 đã cải thiện ECE test của cấu hình cuối. ECE val của I04_256 (0.017985) cũng cao hơn I00 (0.008020), một đánh đổi cần nêu rõ.

[CẦN BỔ SUNG: JSON validation inference, checkpoint/hash, validation prediction và biểu đồ macro-F1–p95 lấy từ phép đo thật; bảng trên hiện chưa đối chiếu được với artifact.]

## 6. Độ trễ và điều kiện triển khai

Theo thông tin được cung cấp, phép đo dùng **Tesla T4**, warmup **10** lần, ít nhất **50** lần lấy mẫu và đồng bộ CUDA trước/sau từng forward. `code/benchmark.py` đo p50/p95/p99 bằng thời gian forward và ghi `includes_preprocessing=False`; vì vậy thời gian đọc ảnh, resize, chuẩn hóa và hậu xử lý không nằm trong số đo này. JSON latency gốc chưa có trong repo để xác nhận cấu hình, số lần lặp và các phân vị.

| Cấu hình | GPU | Dtype | Batch | Kích thước | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Ảnh/s | Warmup | Lượt đo |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| I04_256, một forward | Tesla T4 | FP32 | 1 | 256 | — | ≈ 10.407 | — | — | — | 10 | ≥ 50 |

Nếu JSON xác nhận điều kiện trên, p95 10.407 ms thấp hơn ngưỡng 100 ms của `RUBRIC.md` I5 và cả ngân sách 30 ms/khung; khoảng ngân sách này chỉ xét forward model. Không suy ra FP16 hoặc AMP nhanh hơn FP32 khi chưa xem số đo. Không được lấy p95 đơn view nhân K để thay phép đo TTA thực tế.

[CẦN BỔ SUNG: bảng đầy đủ từ JSON latency gồm p50/p95/p99/mean, throughput, `n`, warmup, GPU, torch version cho batch 1/32 và FP32/AMP/FP16; ghi rõ phương pháp nào dùng độ phân giải 256 và có gộp BN hay không.]

## 7. Cấu hình khóa bằng validation và kết quả test cuối

### 7.1. Cấu hình và validation qua ba seed

F01 được báo cáo là ConvNeXt-Tiny tiền huấn luyện ImageNet, fine-tune toàn bộ, CutMix, chọn checkpoint bằng macro-F1 validation; suy luận `I04_256`, một view, T = 1.0; seed **0, 1, 2**. Notebook đặt 12 epoch làm mặc định, nhưng số epoch thực chạy cần `config.json` final xác nhận. Mốc chung kết `T00` là **ResNet-50 + recipe nền + I00**, cũng chạy ba seed; không đồng nhất với recipe T00 ConvNeXt-Tiny ở phần 4.

| Cấu hình | Seed | Macro-F1 val | Top-1 val | ECE val |
|:---|---:|---:|---:|---:|
| F01 · I04_256 | 0 | 0.976058 | 0.982291 | 0.017985 |
| F01 · I04_256 | 1 | 0.974237 | 0.980577 | 0.018208 |
| F01 · I04_256 | 2 | 0.977861 | 0.982576 | 0.014249 |
| **F01 · mean ± std** | **3 seed** | **0.976052 ± 0.001812** | **0.981815 ± 0.001081** | **0.016814 ± 0.002224** |
| T00 · I00 | 0 | 0.800512 | 0.856612 | 0.011713 |
| T00 · I00 | 1 | 0.805321 | 0.856041 | 0.012829 |
| T00 · I00 | 2 | 0.811528 | 0.860897 | 0.021325 |
| **T00 · mean ± std** | **3 seed** | **0.805787 ± 0.005523** | **0.857850 ± 0.002654** | **0.015289 ± 0.005257** |

Macro-F1 validation trung bình của F01 cao hơn mốc 0.170265. Các std là std mẫu qua ba seed, không phải sai số của từng ảnh. Bảng validation là cơ sở chốt mô hình; các con số test dưới đây không được dùng để đổi quyết định.

### 7.2. Test trên toàn bộ fold 0

Các giá trị dưới đây do người thực hiện cung cấp và được mô tả là tính từ prediction CSV. **Repo hiện không chứa CSV**; vì vậy chưa kiểm tra được mỗi seed có đúng 3.507 ảnh, đủ `Filename, y_true, y_pred, p0…p8`, xác suất chuẩn hóa, nhãn theo subset và chỉ một lượt test. Những điều này phải được kiểm chứng bằng `eval.py score` trước khi nộp.

| Cấu hình | Seed | Macro-F1 test | Top-1 test | Balanced acc. test | ECE test |
|:---|---:|---:|---:|---:|---:|
| F01 | 0 | 0.974507 | 0.980040 | 0.975623 | 0.023167 |
| F01 | 1 | 0.978493 | 0.982891 | 0.979792 | 0.018780 |
| F01 | 2 | 0.979387 | 0.983747 | 0.979945 | 0.016827 |
| **F01 · mean ± std** | **3 seed** | **0.977462 ± 0.002598** | **0.982226 ± 0.001941** | **0.978453 ± 0.002452** | **0.019591 ± 0.003247** |
| T00 | 0 | 0.816573 | 0.863986 | 0.798492 | 0.017917 |
| T00 | 1 | 0.808876 | 0.857428 | 0.764852 | 0.014131 |
| T00 | 2 | 0.809524 | 0.859994 | 0.786471 | 0.021543 |
| **T00 · mean ± std** | **3 seed** | **0.811658 ± 0.004269** | **0.860470 ± 0.003305** | **0.783272 ± 0.017046** | **0.017863 ± 0.003706** |

Theo mean được cung cấp từ các phép tính gốc, F01 tăng khoảng **0.165805 macro-F1** và **0.121756 top-1** so với T00, tương đương 16,58 và 12,18 điểm phần trăm. Std macro-F1 lớn hơn trong hai nhóm là 0.004269; khoảng cách lớn hơn nhiều lần std này và vượt 0.01. Chênh lệch tuyệt đối giữa macro-F1 val và test của F01 là khoảng **0.001410**, dưới ngưỡng 0.02 của rubric. Test cao hơn val một ít; đây là mô tả số học sau khi đã khóa cấu hình, không phải lý do chọn hoặc chạy lại mô hình.

ECE test trung bình F01 0.019591, cao hơn mốc T00 0.017863 theo bảng được cung cấp. Bản final và uncal của F01 được mô tả là giống nhau do T = 1.0; vì vậy **không** thể ghi I4(a) là hiệu chuẩn đã làm giảm ECE test. Không suy ra nhiệt độ 0.9070198983 của I07 sẽ có tác dụng tương tự ở độ phân giải 256 vì cấu hình đó chưa được kiểm chứng trên test.

### 7.3. Hai lớp khó và phân tích lỗi

| Cấu hình / lớp | Precision test, mean ± std | Recall test, mean ± std | F1 test, mean ± std |
|:---|---:|---:|---:|
| F01 · Chinee apple | 0.978968 ± 0.002338 | 0.960177 ± 0.015954 | 0.969428 ± 0.007421 |
| F01 · Snake weed | 0.961035 ± 0.000476 | 0.967320 ± 0.012336 | 0.964143 ± 0.006361 |
| T00 · Chinee apple | 0.938009 ± 0.024761 | 0.489676 ± 0.025546 | 0.643201 ± 0.024751 |
| T00 · Snake weed | 0.762517 ± 0.042938 | 0.712418 ± 0.035462 | 0.735249 ± 0.003916 |

Theo bảng được cung cấp, F01 cải thiện recall của Chinee apple từ 0.489676 lên 0.960177 và Snake weed từ 0.712418 lên 0.967320 so với mốc cùng bài lab. Hai recall F01 vượt mức **88,5%** và **88,8%** của ResNet-50 trong bài báo gốc, như `README.md` mục 2.3 trích dẫn. Đó là mốc tham khảo khác điều kiện huấn luyện, cách tổng hợp và có thể cả cách định nghĩa chỉ số; không coi là so sánh ngang điều kiện. Chưa có prediction hoặc ảnh lỗi để xác định cặp nhầm lẫn nhiều nhất, nên không suy đoán đặc điểm ảnh bị sai.

[CẦN BỔ SUNG: ma trận nhầm lẫn test F01 và T00, số đếm theo hàng nhãn thật/cột nhãn dự đoán từ các prediction CSV, cùng một số ảnh dự đoán sai có Filename và nhãn thật/dự đoán.]

## 8. Đối chiếu các tiêu chí chất lượng trong RUBRIC

| Mục | Bằng chứng hiện được cung cấp | Giới hạn xác minh |
|:---|:---|:---|
| I1 · top-1 test | F01 mean 0.982226 (98,22%), trên mốc tham khảo 95,7%. | Cần prediction đủ ba seed để `eval.py` tính lại. |
| I2 · tăng macro-F1 | Δ ≈ 0.165805, lớn hơn 0.004269 và 0.01. | Cần đối chiếu hai nhóm CSV cùng fold/seed. |
| I3 · lớp khó | Recall F01 0.960177 và 0.967320, trên mốc tham khảo 0.885 và 0.888. | Mốc bài báo khác điều kiện; cần per-class output. |
| I4 · ổn định/hiệu chuẩn | Gap val–test ≈ 0.001410; T final = 1.0. | Có bằng chứng cho ý gap theo số cung cấp; **chưa có** bằng chứng ECE test giảm sau calibration. |
| I5 · thời gian thực | p95 batch 1 FP32 ≈ 10.407 ms trên Tesla T4, dưới 100 ms. | Cần JSON gốc và macro-F1 test/prediction đúng checkpoint để xác nhận. |

Bảng này là đối chiếu điều kiện, **không phải điểm tự chấm chính thức**. Phần I do `eval.py grade` và giảng viên tính lại từ prediction thực tế. Các tiêu chí A–H còn phụ thuộc artifact EDA, config, curve, workbook và khả năng tái lập đã liệt kê trong báo cáo.

## 9. Kết luận và khuyến nghị

Cấu hình cuối được chọn bằng validation là **ConvNeXt-Tiny + CutMix + một view 256 × 256**. Trên bảng ba seed được cung cấp, macro-F1 test cao hơn mốc ResNet-50 + I00 khoảng **0.165805**; mức này lớn hơn std macro-F1 lớn nhất 0.004269. Lựa chọn backbone là thay đổi lớn nhất trong các phép so sánh: từ 0.800512 của B01 lên 0.972956 của B03 trên validation (Δ = 0.172444). Trong cùng backbone, CutMix chỉ nhỉnh hơn recipe T00 khoảng 0.00156 ở một seed, và 256 chỉ nhỉnh hơn I00 khoảng 0.00155. Không cộng các delta này thành đóng góp nhân quả tuyệt đối vì các vòng sàng chủ yếu một seed và có thể khác checkpoint/điều kiện đo.

Nếu triển khai trên robot với ngân sách 30–100 ms/khung, lựa chọn có cơ sở từ số đã cung cấp là ConvNeXt-Tiny single-view 256: p95 forward batch 1 khoảng 10.407 ms trên Tesla T4. Five-crop và TTA hai view cần nhiều forward hơn nên không được chọn khi độ trễ là ưu tiên; cần đo thời gian **toàn pipeline** và phần cứng robot thật trước khi khẳng định đáp ứng chu kỳ cảm biến. Hiệu chuẩn cho I04_256 có thể nghiên cứu thêm bằng validation trong công việc tương lai, nhưng chưa được coi là kết quả test hiện tại.

## 10. Hạn chế và hướng phát triển

Thí nghiệm mới dùng **fold 0**; backbone và ablation chủ yếu **một seed**, còn chung kết mới có ba seed. Fold của DeepWeeds được chia ngẫu nhiên, không theo địa điểm; kết quả có thể lạc quan khi gặp địa điểm, mùa hoặc điều kiện ánh sáng khác. Thời gian huấn luyện khoảng 10–12 epoch trong bảng, ngắn hơn nhiều so với khoảng 100 epoch và augmentation mạnh của bài báo gốc. Ngân sách GPU hạn chế số kết hợp, seed và phép đo đầy đủ. Chưa đánh giá domain shift, chưa có bằng chứng temperature scaling giảm ECE **test** cho cấu hình final; thiếu ảnh lỗi và ma trận nhầm lẫn trong repo.

Việc tiếp theo nên ưu tiên nhiều fold và nhiều seed cho những ablation sát nhau; khớp nhiệt độ cho **chính** I04_256 bằng validation rồi khóa trước một đánh giá mới có quy trình độc lập; thử chưng cất hoặc mô hình nhẹ trên phần cứng biên; đánh giá ảnh mờ, thiếu sáng, mùa/địa điểm khác; và dùng Grad-CAM hoặc attention map sau khi có ảnh dự đoán sai thật. Không mở lại cùng test fold để chọn cấu hình tốt hơn.

## Phụ lục A. Ánh xạ thí nghiệm và khả năng truy vết

| Nhóm | Mã | Ý nghĩa được xác nhận bởi thông tin hiện có | Artifact còn cần |
|:---|:---|:---|:---|
| Backbone | B01–B05 | Lần lượt ResNet-50, ResNeXt-50 32×4d, ConvNeXt-Tiny, DeiT-Small, EfficientNet-B0. | `config.json`, `summary.json`, `history.csv`, curve, tag trọng số. |
| Recipe | T00 | ConvNeXt-Tiny với recipe nền **ở giai đoạn recipe**. | Config/summary để xác nhận seed và mọi field. |
| Recipe | T01–T05, T07–T10 | Chưa có cấu hình thực chạy để gán chính xác thay đổi. | [CẦN BỔ SUNG: cấu hình chính xác của từng Txx từ `config.json`/`summary.json`.] |
| Recipe | T06 | CutMix trên ConvNeXt-Tiny, theo thông tin được cung cấp. | Config/summary và curve. |
| Recipe | T99 | CutMix + focal loss, theo thông tin được cung cấp. | Config/summary và curve. |
| Suy luận | I00, I01, I02, I03 | Single 224; hflip trung bình xác suất; five-crop; hflip trung bình logit. | `inference_record.json`, prediction val. |
| Suy luận | I04_256/288/320, I07, I08 | Single ở ba độ phân giải; temperature scaling; Conv–BN fusion. | `inference_record.json`, latency JSON. |
| Chung kết | F01 | ConvNeXt-Tiny + CutMix + I04_256, seed 0/1/2. | Locked config, checkpoint metadata, prediction/eval output. |
| Mốc chung kết | T00 | ResNet-50 + recipe nền + I00, seed 0/1/2. | Config, prediction/eval output. |

Các mã T01–T10 trong notebook là **kế hoạch** thí nghiệm; chỉ `config.json`/`summary.json` của lần chạy thật mới xác định được thí nghiệm đã thực hiện. Không ghép `T00` ở hai giai đoạn khi tính mean/std hoặc chọn người thắng.

## Phụ lục B. Cấu hình, môi trường và artifact cần bàn giao

- **Final được cung cấp:** `F01`, ConvNeXt-Tiny ImageNet, fine-tune, CutMix, 224 khi train, 256 khi suy luận, một view, T = 1.0, seed 0/1/2; checkpoint theo macro-F1 val. [CẦN BỔ SUNG: `locked_config.json` và ba `config.json` để xác nhận 12 epoch, tag trọng số, batch, LR, alpha CutMix và các field còn lại.]
- **Mốc:** `T00` final, ResNet-50, recipe nền, I00, seed 0/1/2; không dùng recipe T00 ConvNeXt-Tiny để tính delta cuối.
- **Phần cứng/phần mềm được cung cấp:** Tesla T4; Python 3.13.15; torch 2.11.0+cu128; torchvision 0.26.0+cu128; AMP bật. [CẦN BỔ SUNG: version `timm`, `fvcore`, `pandas`, CUDA driver/runtime và tag trọng số từ artifact.]
- **Notebook chạy lại:** mã nguồn tại [`code/lab_day2.ipynb`](code/lab_day2.ipynb). [CẦN BỔ SUNG: link notebook Colab/Kaggle đã chạy và chỉ dẫn truy cập artifact bền vững.]
- **Gói minh chứng còn thiếu trong repo:** `results.xlsx`; các `summary.json`, `config.json`, `history.csv`; JSON validation inference và latency; prediction F01/T00 seed 0/1/2 với `Filename, y_true, y_pred, p0…p8`; output `eval.py score`/`grade`; curves cho B/T/F; biểu đồ EDA; ma trận nhầm lẫn và ảnh lỗi. Chỉ chèn hình bằng đường dẫn tương đối khi file thật đã có.

**Lưu ý làm tròn:** một số mean/std do yêu cầu cung cấp khác phép tính từ **ba số seed đã làm tròn sáu chữ số** ở chữ số cuối: T00 top-1 test cho 0.860469 khi tính lại từ số hiển thị, thay vì 0.860470 được cung cấp; T00 balanced-accuracy std cho 0.017047 thay vì 0.017046; T00 ECE mean cho 0.017864 thay vì 0.017863. Theo mean hiển thị, Δ macro-F1 test là 0.165804 thay vì 0.165805 được cung cấp; Δ T06 − recipe T00 là 0.001557 thay vì khoảng 0.001556. Báo cáo giữ các số tổng hợp được cung cấp; sai khác khoảng 0.000001 có thể do làm tròn đầu vào nhưng **chưa thể xác minh nguyên nhân** khi thiếu artifact gốc. Trước khi nộp, cần ưu tiên phép tính lại từ prediction/log chưa làm tròn và ghi lại nếu kết quả khác.
