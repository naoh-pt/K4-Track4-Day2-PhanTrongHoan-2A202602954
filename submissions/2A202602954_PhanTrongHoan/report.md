# Báo cáo Lab Day 2: phân loại DeepWeeds

## 1. Tóm tắt

DeepWeeds là bài toán phân loại 17.509 ảnh RGB thành chín lớp. Trên fold 0, tôi so sánh năm backbone, 12 công thức huấn luyện và chín phương pháp suy luận/hiệu chuẩn. Cấu hình cuối được chọn bằng **validation**: ConvNeXt-Tiny khởi tạo ImageNet, fine-tune toàn bộ, CutMix, suy luận một view 256 × 256 (`I04_256`). Tính từ sáu prediction test CSV, ba seed của F01 đạt macro-F1 **0.977462 ± 0.002598** và top-1 **0.982226 ± 0.001941**. So với mốc ResNet-50 `T00`, mức tăng lần lượt **0.165805** và **0.121756**, lớn hơn rõ rệt độ lệch chuẩn giữa seed. Nhiệt độ final là **T = 1.0**; kết quả test không hưởng lợi từ temperature scaling.

## 2. Dữ liệu và thiết lập

### 2.1. Split và phân bố lớp

DeepWeeds gồm tám loài cỏ dại và lớp `Negative`, ảnh gốc 256 × 256. Tôi dùng đúng fold 0 của tác giả: train **10.501**, validation **3.501**, test **3.507** ảnh; không chia lại. [Log Version 1](../../report_inputs/version_1/track4-day2.log) ghi giao tên file của ba cặp tập bằng 0 và hợp ba tập bằng **17.509**.

| Label / lớp | Train | Validation | Test |
|:---|---:|---:|---:|
| 0 · Chinee Apple | 675 | 225 | 226 |
| 1 · Lantana | 637 | 213 | 213 |
| 2 · Parkinsonia | 618 | 206 | 207 |
| 3 · Parthenium | 613 | 204 | 205 |
| 4 · Prickly Acacia | 637 | 212 | 213 |
| 5 · Rubber Vine | 605 | 202 | 202 |
| 6 · Siam Weed | 644 | 215 | 215 |
| 7 · Snake Weed | 609 | 203 | 204 |
| 8 · Negatives | 5.463 | 1.821 | 1.822 |
| **Tổng** | **10.501** | **3.501** | **3.507** |

Lớp `Negative` chiếm khoảng 52%; tỷ lệ lớn nhất/nhỏ nhất lần lượt **9.0298**, **9.0149**, **9.0198** trên train/val/test. Vì top-1 có thể bị lớp này chi phối, tôi dùng **macro-F1** làm chỉ số chọn chính; top-1, balanced accuracy, ECE 15 bin và chỉ số từng lớp được báo cáo thêm theo [`eval.py`](../../eval.py). Tổng theo Label của subset có 1.126 ảnh lớp 0 và 1.063 ảnh lớp 1, khác metadata gốc một ảnh do bất nhất được ghi ở mục 2.2. Không sửa subset để ép khớp Table 1 của bài báo.

Gói artifact không có ảnh EDA; còn cần biểu đồ phân bố lớp và lưới 27 ảnh thật. Không tạo ảnh mẫu giả.

### 2.2. Sanity check và ranh giới dữ liệu

Kết quả sanity và kiểm thử được người thực hiện cung cấp ở lần bàn giao trước: batch `[8, 3, 224, 224]` kiểu `float32`, khoảng chuẩn hóa xấp xỉ −2.1179 đến 2.6400; CE ban đầu **2.118345** gần ln(9) = 2.197225. Overfit một batch trong tối đa 80 bước đưa CE xuống **5.9604 × 10⁻⁶**; kiểm thử đạt **75 pytest** và **38 unittest**. Gói `report_inputs/` hiện không có output chi tiết của các phép kiểm này để đối chiếu lại. Đây là kiểm tra kỹ thuật, không phải điểm mô hình chính.

Metadata `labels.csv` và `train_subset0.csv` lệch đúng một Filename: `20170714-110407-3.jpg` mang Label 1/Species Lantana trong metadata, Label 0 trong subset. Pipeline giữ Label subset làm target và phát `RuntimeWarning`; CSV không được sửa. Train chỉ cập nhật trọng số; validation chọn backbone, recipe, checkpoint và suy luận. Test chỉ dùng sau khi khóa cấu hình trong [`locked_config.json`](../../report_inputs/version_8/deepweeds_outputs/final/locked_config.json), không dùng để chọn lại.

### 2.3. Môi trường và công thức nền

Notebook chạy trên Kaggle với **Tesla T4**. Log ghi Python **3.13.15**, PyTorch **2.11.0+cu128**, torchvision **0.26.0+cu128**, timm **1.0.29**, fvcore **0.1.5.post20221221**. `summary.json` xác nhận fine-tune khởi tạo ImageNet, 12 epoch, batch 64, ảnh train 224, basic augmentation, CE, AdamW, LR backbone **1e-4**, LR head **1e-3**, weight decay **0.05**, warmup 1 epoch rồi cosine, AMP bật; checkpoint theo macro-F1 validation. Artifact ghi `init=finetune` nhưng **không ghi tag pretrained cụ thể** của từng backbone. Config thực chạy được nhúng trong summary ở Version 2–4 và 8.

## 3. So sánh backbone trên validation

Năm `summary.json` của Version 2 dùng cùng seed 0 và recipe nền. Sheet `Backbones` trong [`results.xlsx`](results.xlsx) chứa nguồn từng dòng. Không có latency sơ bộ riêng cho B01–B05; latency ở mục 6 chỉ của cấu hình I04_256.

| Exp | Backbone | Params M | GMAC | Macro-F1 val | Top-1 val | Giây/epoch |
|:---|:---|---:|---:|---:|---:|---:|
| B01 | ResNet-50 | 23.526473 | 4.087155 | 0.800512 | 0.856612 | 57.395 |
| B02 | ResNeXt-50 32×4d | 22.998345 | 4.228450 | 0.748919 | 0.793773 | 77.833 |
| **B03** | **ConvNeXt-Tiny** | **27.827049** | **4.454770** | **0.972956** | **0.979720** | **70.036** |
| B04 | DeiT-Small/16-224 | 21.669129 | 4.240838 | 0.948585 | 0.963439 | 48.876 |
| B05 | EfficientNet-B0 | 4.019077 | 0.384546 | 0.752492 | 0.819480 | 37.042 |

ConvNeXt-Tiny đạt macro-F1 validation cao nhất, hơn B01 **0.172444**. DeiT-Small đứng thứ hai và nhanh hơn B03 theo giây/epoch trong lần đo. EfficientNet-B0 nhỏ, ít GMAC và có giây/epoch thấp nhất nhưng chất lượng thấp hơn nhiều. GMAC không dự đoán hoàn toàn thời gian thực thi. Đây là sàng lọc **một seed**, nên chưa khẳng định được các chênh lệch rất nhỏ. Chưa có curves B01–B05 hoặc latency từng backbone để vẽ đầy đủ quan hệ chất lượng–độ trễ.

## 4. Công thức huấn luyện trên ConvNeXt-Tiny

`T00` **recipe Version 3** dùng ConvNeXt-Tiny, khác `T00` **baseline final Version 8–9** dùng ResNet-50 + `I00`. Tôi đối chiếu `config` nhúng trong JSON của từng run, không suy từ mã exp. Bảng sau là validation seed 0; Δ tính với recipe T00 từ số chưa làm tròn.

| Exp | Thay đổi so với recipe T00 | Best epoch | Macro-F1 val | Δ F1 | Top-1 val | Giây/epoch |
|:---|:---|---:|---:|---:|---:|---:|
| T00 | Nền: fine-tune/basic/CE | 12 | 0.972956 | 0.000000 | 0.979720 | 69.443 |
| T01 | Frozen backbone | 11 | 0.851600 | −0.121356 | 0.882319 | 42.973 |
| T02 | Scratch, không pretrained | 12 | 0.305518 | −0.667438 | 0.544702 | 68.643 |
| T03 | Color augmentation | 12 | 0.968189 | −0.004767 | 0.976292 | 78.259 |
| T04 | RandAugment | 10 | 0.973224 | +0.000268 | 0.979149 | 68.372 |
| T05 | Mixup | 10 | 0.965909 | −0.007048 | 0.973436 | 69.422 |
| **T06** | **CutMix, alpha 1.0** | **10** | **0.974513** | **+0.001556** | **0.980863** | **69.040** |
| T07 | Label smoothing, 0.1 | 11 | 0.969879 | −0.003078 | 0.976292 | 68.766 |
| T08 | Focal loss, gamma 2 | 10 | 0.973130 | +0.000174 | 0.980577 | 67.512 |
| T09 | Weighted CE | 12 | 0.964355 | −0.008602 | 0.972008 | 68.023 |
| T10 | EMA, decay 0.999 | 12 | 0.968571 | −0.004385 | 0.975721 | 68.721 |
| T99 | CutMix + focal | 10 | 0.971149 | −0.001807 | 0.977721 | 81.430 |

T06 cao nhất vòng recipe nhưng chỉ hơn T00 **0.001556** ở một seed; chưa đủ cơ sở khẳng định CutMix ổn định tốt hơn. T99 thấp hơn T06, nên hai kỹ thuật không tạo hiệu ứng cộng dồn trong lần thử này. T02 giảm rõ rệt khi bỏ khởi tạo ImageNet trong cùng ngân sách 12 epoch; không quy mọi chênh lệch cho riêng một yếu tố ngoài config. T01 cũng giảm khi đóng băng backbone. Gói không có `history.csv`/curves B/T, nên không suy đoán tốc độ hội tụ hay quá khớp theo epoch.

## 5. Phương pháp suy luận trên validation

Version 6 dùng cùng checkpoint T06 seed 0, SHA-256 `729688896c24a584581e1982824dbaed601cf363010e8be9f6e0ac24405d0d8c`. K là số forward/view. I07 hiệu chuẩn single-view 224, **không phải** I04_256.

| Phương pháp | K | Macro-F1 val | Top-1 val | ECE val |
|:---|---:|---:|---:|---:|
| I00 · single 224 | 1 | 0.974513 | 0.980863 | 0.008020 |
| I01 · hflip, mean probability | 2 | 0.975133 | 0.981434 | 0.008276 |
| I02 · five-crop | 5 | 0.974889 | 0.981434 | 0.009220 |
| I03 · hflip, mean logit | 2 | 0.975133 | 0.981434 | 0.008212 |
| **I04_256 · single 256** | **1** | **0.976058** | **0.982291** | **0.017985** |
| I04_288 · single 288 | 1 | 0.972385 | 0.978292 | 0.032886 |
| I04_320 · single 320 | 1 | 0.967094 | 0.972579 | 0.042501 |
| I07 · T = 0.907020 trên I00 | 1 | 0.974513 | 0.980863 | 0.002411 |
| I08 · Conv–BN fusion trên I00 | 1 | 0.974513 | 0.980863 | 0.008020 |

I04_256 đứng đầu validation; 288 và 320 làm điểm giảm. Hflip tăng khoảng **0.000620** so với I00 nhưng cần hai forward; five-crop cần năm forward và không vượt single 256. I07 giảm ECE validation từ **0.008020** xuống **0.002411**, không đổi macro-F1; final vẫn dùng **T = 1.0**. I08 giữ nguyên metric, nhưng log báo **0 cặp Conv–BN được gộp**, nên không suy ra tăng tốc. Không có latency TTA/five-crop thực đo; K không được dùng thay số đo.

## 6. Độ trễ trên Tesla T4

Sáu JSON Version 7 đo `I04_256`, input tạo ngoài khoảng đo, mô hình `eval`, đồng bộ CUDA trước/sau forward. Notebook/log ghi **10 warmup**; JSON ghi **n=50**, `includes_preprocessing=false`, `fused_bn_pairs=0`, torch `2.11.0+cu128`, `official=true`. Mọi số thời gian là mili giây; nguồn mỗi hàng trong sheet `Latency`.

| Batch | Dtype | p50 ms | p95 ms | p99 ms | Mean ms | Ảnh/s |
|---:|:---|---:|---:|---:|---:|---:|
| 1 | FP32 | 6.039904 | 10.407006 | 10.522778 | 7.167530 | 165.565545 |
| 1 | AMP | 8.527171 | 9.094612 | 9.345803 | 8.625350 | 117.272188 |
| 1 | FP16 | 6.023711 | 6.361301 | 7.425813 | 6.085339 | 166.010634 |
| 32 | FP32 | 147.047348 | 149.301843 | 149.819785 | 147.113235 | 217.616981 |
| 32 | AMP | 57.931148 | 58.450201 | 58.876059 | 57.945183 | 552.379870 |
| 32 | FP16 | 46.213007 | 47.334002 | 47.755597 | 46.204667 | 692.445751 |

Batch 1 FP32 p95 **10.407006 ms** thấp hơn mốc 100 ms của rubric và ngân sách 30 ms/khung **cho riêng model forward trên T4**. AMP có p50 batch 1 cao hơn FP32, còn FP16 có p95 thấp hơn trong phép đo này. Batch 32 AMP/FP16 tăng thông lượng so với FP32; không đồng nhất thông lượng batch 32 với độ trễ một khung. Chưa đo camera, giải mã ảnh, preprocessing hoặc phần cứng robot.

## 7. Cấu hình cuối và kết quả test

### 7.1. Khóa cấu hình và validation

[`locked_config.json`](../../report_inputs/version_8/deepweeds_outputs/final/locked_config.json) chốt ConvNeXt-Tiny + CutMix + `I04_256`, seed 0/1/2. `summary.json` final xác nhận 12 epoch, train 224, batch 64, fine-tune, AMP, checkpoint theo macro-F1 val. Mốc final `T00` là ResNet-50 + công thức nền + `I00`, cũng ba seed. Bảng là **validation selected inference record Version 8**, không phải test.

| Cấu hình | Seed | Macro-F1 val | Top-1 val | ECE val |
|:---|---:|---:|---:|---:|
| F01 · I04_256 | 0 | 0.976058 | 0.982291 | 0.017985 |
| F01 · I04_256 | 1 | 0.974237 | 0.980577 | 0.018208 |
| F01 · I04_256 | 2 | 0.977861 | 0.982576 | 0.014249 |
| **F01 mean ± std** | **3** | **0.976052 ± 0.001812** | **0.981815 ± 0.001081** | **0.016814 ± 0.002224** |
| T00 · I00 | 0 | 0.800512 | 0.856612 | 0.011713 |
| T00 · I00 | 1 | 0.805321 | 0.856041 | 0.012829 |
| T00 · I00 | 2 | 0.811528 | 0.860897 | 0.021325 |
| **T00 mean ± std** | **3** | **0.805787 ± 0.005523** | **0.857850 ± 0.002654** | **0.015289 ± 0.005257** |

F01 hơn mốc **0.170265 macro-F1 validation** theo mean ba seed; đây là cơ sở chốt, không dùng test để đổi quyết định. Có sáu curves final: [F01 seed 0](curves/F01_seed0.png), [1](curves/F01_seed1.png), [2](curves/F01_seed2.png); [T00 seed 0](curves/T00_seed0.png), [1](curves/T00_seed1.png), [2](curves/T00_seed2.png). Chưa có history.csv để đối chiếu chi tiết từng điểm epoch.

### 7.2. Test độc lập từ prediction CSV

Sáu CSV `F01/T00 × seed 0/1/2` đều có **3.507** dòng, đủ `Filename, y_true, y_pred, p0…p8`, Filename duy nhất, xác suất hữu hạn không âm tổng gần 1, `y_pred = argmax(p)`; Filename và `y_true` cùng thứ tự giữa seed. ZIP Version 9 có sáu marker `completed.json`, prediction trùng byte với `predictions/`. Chỉ số tính theo công thức của `eval.py`, ECE 15 bin; mean/std mẫu `ddof=1`. Nguồn đầy đủ: [`final_metrics.csv`](../../report_assets/final_metrics.csv).

| Cấu hình | Seed | Macro-F1 test | Top-1 test | Balanced acc. | ECE test |
|:---|---:|---:|---:|---:|---:|
| F01 | 0 | 0.974507 | 0.980040 | 0.975623 | 0.023167 |
| F01 | 1 | 0.978493 | 0.982891 | 0.979792 | 0.018780 |
| F01 | 2 | 0.979387 | 0.983747 | 0.979945 | 0.016827 |
| **F01 mean ± std** | **3** | **0.977462 ± 0.002598** | **0.982226 ± 0.001941** | **0.978453 ± 0.002452** | **0.019591 ± 0.003247** |
| T00 | 0 | 0.816573 | 0.863986 | 0.798492 | 0.017917 |
| T00 | 1 | 0.808876 | 0.857428 | 0.764852 | 0.014130 |
| T00 | 2 | 0.809524 | 0.859994 | 0.786471 | 0.021543 |
| **T00 mean ± std** | **3** | **0.811658 ± 0.004269** | **0.860470 ± 0.003305** | **0.783272 ± 0.017046** | **0.017863 ± 0.003706** |

Từ số chưa làm tròn, F01 hơn T00 **0.165805 macro-F1** và **0.121756 top-1**; std macro-F1 lớn nhất chỉ **0.004269**. Gap `|macro-F1 val − test|` F01 là **0.001410**; test cao hơn val một ít nhưng không phải lý do chọn cấu hình. ECE test F01 **0.019591** cao hơn T00 **0.017863**. File final và uncal giống byte vì **T = 1.0**; không có bằng chứng temperature scaling giảm ECE test final. Gói chưa có labels fold 0 gốc, nên chỉ đối chiếu `y_true` trong prediction; chưa chạy được nguyên `eval.py score/grade` với subset chính thức.

### 7.3. Ma trận nhầm lẫn và lỗi theo lớp

Hai ma trận cộng gộp ba seed, mỗi seed dự đoán cùng 3.507 ảnh. Tổng 10.521 là **lượt dự đoán**, không phải số ảnh độc lập; hàng là nhãn thật, cột là nhãn dự đoán.

![Ma trận nhầm lẫn F01](../../report_assets/confusion_matrix_F01.png)

![Ma trận nhầm lẫn T00](../../report_assets/confusion_matrix_T00.png)

Bảng hai lớp khó dưới là mean ± std metric từng seed; đủ chín lớp ở Phụ lục C và [`per_class_metrics.csv`](../../report_assets/per_class_metrics.csv).

| Cấu hình / lớp | Precision | Recall | F1 |
|:---|---:|---:|---:|
| F01 · Chinee Apple | 0.978968 ± 0.002338 | 0.960177 ± 0.015954 | 0.969428 ± 0.007421 |
| F01 · Snake Weed | 0.961035 ± 0.000476 | 0.967320 ± 0.012336 | 0.964143 ± 0.006361 |
| T00 · Chinee Apple | 0.938009 ± 0.024761 | 0.489676 ± 0.025546 | 0.643201 ± 0.024751 |
| T00 · Snake Weed | 0.762517 ± 0.042938 | 0.712418 ± 0.035462 | 0.735249 ± 0.003916 |

Recall F01 của Chinee Apple và Snake Weed là **96,02%** và **96,73%**, trên mốc **88,5%** và **88,8%** mà [`README.md`](README.md) trích từ [Olsen et al., 2019](https://doi.org/10.1038/s41598-018-38343-3). Đây chỉ là đối chiếu tham khảo vì điều kiện huấn luyện, cách tổng hợp và số fold khác. Chưa có ảnh gốc của các Filename dự đoán sai; không suy đoán đặc điểm thị giác hay nguyên nhân lỗi.

## 8. Đối chiếu RUBRIC

| Mục | Bằng chứng | Giới hạn |
|:---|:---|:---|
| I1 · top-1 | F01 mean **98,22%**, trên mốc tham khảo 95,7%. | Từ prediction; chưa đối chiếu nhãn gốc bằng `eval.py`. |
| I2 · macro-F1 | Δ **0.165805** > std lớn nhất **0.004269** và > 0.01. | Mốc T00 final cùng ba seed. |
| I3 · lớp khó | Recall **0.960177** và **0.967320**. | Mốc bài báo khác điều kiện. |
| I4 · ổn định/hiệu chuẩn | Gap val–test **0.001410**; final T = 1.0. | Không chứng minh ECE test giảm sau hiệu chuẩn. |
| I5 · latency | p95 batch 1 FP32 **10.407006 ms** trên T4. | Chỉ model forward, không gồm preprocessing. |

Bảng là đối chiếu số học, không phải điểm tự chấm chính thức. Giảng viên có thể tính lại từ nhãn gốc. Tiêu chí báo cáo G có bảng, ma trận và phần hạn chế; phần ảnh lỗi cần bổ sung khi có ảnh gốc.

## 9. Kết luận và khuyến nghị

Cấu hình tốt nhất theo validation là **ConvNeXt-Tiny + CutMix + một view 256**. Macro-F1 test hơn mốc ResNet-50 + I00 **0.165805**, lớn hơn nhiễu giữa seed. Thay backbone là thay đổi lớn nhất trong các vòng sàng: B01 **0.800512** lên B03 **0.972956** macro-F1 val, Δ **0.172444**. T06 chỉ hơn recipe T00 **0.001556**, còn I04_256 hơn I00 **0.001546** trong một seed; không cộng thẳng các delta để suy quan hệ nhân quả tuyệt đối.

Với ngân sách robot 30–100 ms/khung, tôi chọn single-view 256 thay vì TTA nhiều view: p95 forward FP32 batch 1 trên T4 **10.407006 ms**. Cần đo toàn pipeline và phần cứng robot thật trước khi cam kết thời gian thực. Nếu ưu tiên xác suất được hiệu chuẩn, có thể nghiên cứu temperature scaling **riêng cho I04_256 trên validation** trong một giao thức mới; kết hợp đó chưa được kiểm chứng trên final test.

## 10. Hạn chế và hướng phát triển

Chỉ dùng **fold 0**; backbone, recipe và inference chủ yếu chạy **một seed**, final mới có ba seed. Split DeepWeeds là ngẫu nhiên, không theo địa điểm; test có thể lạc quan khi gặp mùa/vùng khác. 12 epoch ngắn hơn khoảng 100 epoch trong [bài báo gốc](https://doi.org/10.1038/s41598-018-38343-3). Ngân sách GPU hạn chế số seed và số kết hợp. Chưa chứng minh calibration cải thiện ECE test final; chưa đánh giá domain shift, ảnh mờ/thiếu sáng, hoặc thời gian toàn pipeline. Thiếu ảnh lỗi, ảnh EDA và curves B/T để phân tích sâu.

Việc tiếp theo là nhiều fold và seed cho ablation gần nhau, thử mô hình nhẹ/chưng cất cho phần cứng biên, kiểm tra ảnh theo mùa/địa điểm, và dùng Grad-CAM/attention map khi có ảnh lỗi thật. Nếu thử nhiệt độ cho 256, chỉ fit bằng validation, khóa trước đánh giá độc lập mới; không chạy lại cùng test để chọn kết quả tốt hơn.

## Phụ lục A. Ánh xạ version và khử trùng

| Version | Nội dung chính thức | Cách dùng |
|---:|:---|:---|
| 1 | Baseline ban đầu T00 ResNet-50 seed 0 | EDA/sanity và đối chiếu; không ghép với recipe T00. |
| 2 | B01–B05 | Năm backbone. |
| 3 | Recipe T00–T10 | Config/metric ablation thực chạy. |
| 4 | T99 | CutMix + focal. |
| 6 | I00–I08 | Chín inference record trên T06 seed 0. |
| 7 | Sáu latency JSON | I04_256 trên T4. |
| 8 | F01/T00 ba seed và validation đã chọn | Locked config, summary, inference record. |
| 9 | Final test | `test_attempts.zip`, completed markers, prediction. |
| 10 | Xuất `results.xlsx` | Không phải thí nghiệm mới. |

Version 5 không tồn tại vì lần chạy hủy. JSON trong Version 9/10 là bản restore **trùng SHA-256** với Version 8, không đếm thêm; prediction trong ZIP trùng byte với file `predictions/`. Tôi phân biệt theo loại artifact, scope/version, exp_id, seed, checkpoint hash và nội dung. Version 1 và B01 có cùng macro-F1 val nhưng giây/epoch khác (khoảng 50.904 và 57.395); đó là hai lần ghi ở hai giai đoạn, không lấy trung bình thời gian. Không thấy mâu thuẫn metric giữa các bản sao Version 8–10. Kiểm kê ở [`artifact_inventory.json`](../../report_assets/artifact_inventory.json).

## Phụ lục B. Cấu hình và khả năng tái lập

- **F01:** `convnext_tiny`, fine-tune ImageNet, CutMix alpha 1.0, CE, train 224, test `I04_256` một view, T = 1.0, 12 epoch, batch 64, seed 0/1/2; checkpoint theo macro-F1 val.
- **T00 final:** `resnet50`, fine-tune ImageNet, basic + CE, train 224, test `I00`, seed 0/1/2. Không đồng nhất với recipe T00 ConvNeXt-Tiny Version 3.
- **Recipe:** T01 frozen; T02 scratch; T03 color; T04 randaug; T05 mixup; T06 cutmix; T07 label smoothing; T08 focal; T09 weighted CE; T10 EMA; T99 cutmix + focal. Từng field và JSON nguồn trong sheet `Training`.
- **Notebook đã chạy thực tế (Kaggle Version 9):** [Kaggle Notebook trhphan/track4-day2 (Version 9, scriptVersionId=355337320)](https://www.kaggle.com/code/trhphan/track4-day2?scriptVersionId=355337320). Mã nguồn tương đương được lưu tại [`code/lab_day2.ipynb`](code/lab_day2.ipynb). Hướng dẫn chi tiết thứ tự chạy, các cờ điều khiển, cách chạy `eval.py` và tái lập từng bước xem tại [`SUBMISSION_README.md`](README.md).
- **Còn thiếu:** CSV nhãn fold 0 để chạy nguyên `eval.py score/grade`; `history.csv`, `config.json` rời, curves B/T, ảnh EDA, ảnh dự đoán sai, tag pretrained chính xác. Config thực chạy đã có dạng nhúng trong summary, nên T01–T10 không phải phỏng đoán.

## Phụ lục C. Chỉ số chín lớp trên test

Mỗi ô là **mean ± sample std** qua ba seed; support là số ảnh mỗi seed. Bảng đầy đủ precision/recall/F1 của **cả F01 và T00** nằm ở [`per_class_metrics.csv`](../../report_assets/per_class_metrics.csv) và sheet `PerClass`.

| Lớp | Support | F01 precision | F01 recall | F01 F1 | T00 precision | T00 recall | T00 F1 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Chinee Apple | 226 | 0.978968 ± 0.002338 | 0.960177 ± 0.015954 | 0.969428 ± 0.007421 | 0.938009 ± 0.024761 | 0.489676 ± 0.025546 | 0.643201 ± 0.024751 |
| Lantana | 213 | 0.975389 ± 0.018804 | 0.982786 ± 0.002711 | 0.979016 ± 0.009956 | 0.875630 ± 0.027178 | 0.813772 ± 0.054414 | 0.842246 ± 0.019117 |
| Parkinsonia | 207 | 0.979334 ± 0.007130 | 0.990338 ± 0.000000 | 0.984797 ± 0.003609 | 0.910051 ± 0.036677 | 0.896940 ± 0.019524 | 0.902926 ± 0.012448 |
| Parthenium | 205 | 0.985389 ± 0.004783 | 0.985366 ± 0.004878 | 0.985366 ± 0.002434 | 0.893567 ± 0.022307 | 0.695935 ± 0.015681 | 0.782463 ± 0.018432 |
| Prickly Acacia | 213 | 0.955244 ± 0.008519 | 0.965571 ± 0.014343 | 0.960289 ± 0.002795 | 0.727305 ± 0.039652 | 0.838811 ± 0.054414 | 0.777390 ± 0.013466 |
| Rubber Vine | 202 | 0.980353 ± 0.012525 | 0.978548 ± 0.010305 | 0.979365 ± 0.002762 | 0.931161 ± 0.015580 | 0.777228 ± 0.019802 | 0.847059 ± 0.009547 |
| Siam Weed | 215 | 0.984561 ± 0.002741 | 0.989147 ± 0.007105 | 0.986845 ± 0.004863 | 0.860718 ± 0.007382 | 0.871318 ± 0.011705 | 0.865932 ± 0.005062 |
| Snake Weed | 204 | 0.961035 ± 0.000476 | 0.967320 ± 0.012336 | 0.964143 ± 0.006361 | 0.762517 ± 0.042938 | 0.712418 ± 0.035462 | 0.735249 ± 0.003916 |
| Negatives | 1.822 | 0.988999 ± 0.000958 | 0.986828 ± 0.001647 | 0.987912 ± 0.001263 | 0.867954 ± 0.019280 | 0.953348 ± 0.013184 | 0.908453 ± 0.004728 |

F01 có F1 trung bình cao hơn T00 ở cả chín lớp trong cùng test. Chênh lớn nhất thuộc Chinee Apple; đây là phân tích kết quả cuối, không dùng để chọn lại mô hình.
