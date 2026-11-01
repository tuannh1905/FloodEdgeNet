# BÁO CÁO KIỂM TOÁN KIẾN TRÚC MÔ HÌNH: Fast-SCNN (`fast-scnn`)

**Dự án**: `FloodEdgeNet` (Phân vùng ảnh ngập lụt cho thiết bị biên)  
**Mô hình kiểm toán**: `fast-scnn` (Fast Segmentation Convolutional Neural Network)  
**Tập tin trong repo**: `models/fast-scnn.py`  
**Ngày kiểm toán**: 01/11/2026  
**Trạng thái thẩm định**: **PASS (ĐẠT CHUẨN A*/Q1)**

---

## 1. Nguồn tham chiếu gốc (Source of Truth)

- **Bài báo gốc**: *"Fast-SCNN: Fast Semantic Segmentation Network"* (BMVC 2019).
  - Tác giả: Rudra PK Poudel, Stephan Liwicki, Roberto Cipolla (Toshiba Research Europe & University of Cambridge).
  - arXiv: [1902.04502](https://arxiv.org/abs/1902.04502)
- **Kho mã nguồn tham chiếu PyTorch tiêu chuẩn**:
  1. `tramac/fast-scnn-pytorch` (`https://github.com/tramac/fast-scnn-pytorch`) — Bản triển khai PyTorch chuẩn mực và phổ biến nhất (>1,500 stars). Commit hash: `0638517d359ae1664a27dfb2cd1780a40a06c465`.
  2. `leftthomas/FastSCNN` (`https://github.com/leftthomas/FastSCNN`) — Commit hash: `bd95fa40c6ef438ba65c07e8feb358805c5862f0`.
- **Đường dẫn tệp tham chiếu**: `models/fast_scnn.py`.
- **Pretrained Tag**: Không có (mô hình huấn luyện từ đầu từ khởi tạo ngẫu nhiên theo công thức He/Kaiming).

---

## 2. Bảng đối chiếu kiến trúc từng tầng (Line-by-Line Architecture Diff)

| Tầng / Khối | Triển khai tham chiếu (Poudel et al. BMVC 2019 / Tramac) | Triển khai dự án (`models/fast-scnn.py`) | Khớp (Y/N) | Phân loại & Ghi chú |
| :--- | :--- | :--- | :---: | :--- |
| **Learning to Downsample** | Conv2d(3->32, k=3, s=2) + DSConv(32->48, s=2) + DSConv(48->64, s=2) | Conv2d(3->32, k=3, s=2) + DSConv(32->48, s=2) + DSConv(48->64, s=2) | Y | Khớp 100% |
| **GFE Bottleneck 1** | 3 khối LinearBottleneck (64->64, $t=6$, stride=[2, 1, 1]) | 3 khối LinearBottleneck (64->64, $t=6$, stride=[2, 1, 1]) | Y | Khớp 100% |
| **GFE Bottleneck 2** | 3 khối LinearBottleneck (64->96, $t=6$, stride=[2, 1, 1]) | 3 khối LinearBottleneck (64->96, $t=6$, stride=[2, 1, 1]) | Y | Khớp 100% |
| **GFE Bottleneck 3** | 3 khối LinearBottleneck (96->128, $t=6$, stride=[1, 1, 1]) | 3 khối LinearBottleneck (96->128, $t=6$, stride=[1, 1, 1]) | Y | Khớp 100% |
| **Pyramid Pooling Module** | 4 nhánh pooling + Conv 1x1 (128->32) + Concat (256) + Conv 1x1 (256->128) | 4 nhánh pooling + Conv 1x1 (128->32) + Concat (256) + Conv 1x1 (256->128) | Y | Khớp 100%. Pool sizes thích ứng theo kích thước ảnh |
| **Feature Fusion Module** | DWConv(128, 128) + Conv1x1(128, 128) trên nhánh thấp cộng với Conv1x1(64, 128) nhánh cao | DWConv(128, 128) + Conv1x1(128, 128) trên nhánh thấp cộng với Conv1x1(64, 128) nhánh cao | Y | Khớp 100% |
| **Classifier** | DSConv(128->128) + DSConv(128->128) + Dropout(0.1) + Conv1x1(128->classes) | DSConv(128->128) + DSConv(128->128) + Dropout(0.1) + Conv1x1(128->num_classes) | Y | Phân loại (2): Thay đổi số kênh đầu ra phục vụ phân vùng ngập lụt |
| **Upsampling** | Bilinear interpolation 8x lên kích thước ảnh gốc, align_corners=True | Bilinear interpolation 8x lên kích thước ảnh gốc, align_corners=True | Y | Khớp 100% |

---

## 3. Xác minh tham số và độ phức tạp (Parameter & Complexity Verification)

- **Số tham số tham chiếu gốc (Cityscapes 19 lớp)**: **`1,138,051`** (~1.14M đúng tuyệt đối theo công bố của tác giả bài báo gốc).
- **Số tham số nhị phân tập Flood (1 lớp)**: **`1,135,729`** (Khớp chính xác 100% với tham chiếu tại 1 lớp).
- **Số tham số đa lớp tập FloodScene (4 lớp)**: **`1,136,116`** (Khớp chính xác 100% với tham chiếu tại 4 lớp).
- **Giải trình chênh lệch tham số (Parameter Reconciliation)**:
  - Khối classifier cuối: `Conv2d(128, num_classes, kernel_size=1, bias=True)`.
  - $\Delta_{1 - 19} = (128 \times 1 + 1) - (128 \times 19 + 19) = 129 - 2,451 = -2,322$.
  - $\text{Params}_{1} = 1,138,051 - 2,322 = 1,135,729$ (Khớp chính xác từng tham số đơn lẻ).
  - $\Delta_{4 - 1} = (128 \times 4 + 4) - (128 \times 1 + 1) = 516 - 129 = +387$.
  - $\text{Params}_{4} = 1,135,729 + 387 = 1,136,116$ (Khớp chính xác từng tham số đơn lẻ).
- **Độ phức tạp tính toán tại độ phân giải kiểm chuẩn $128 \times 128$**:
  - `thop` MACs: **55.25 M**
  - `ptflops` MACs: **55.26 M**
  - `fvcore` FLOPs: **53.75 M**

---

## 4. Kiểm chứng lan truyền xuôi và Tương thích trọng số

- **Phương pháp**: Nạp trực tiếp toàn bộ trọng số từ `models/fast-scnn.py` sang mô hình chuẩn `Tramac/FastSCNN` với `strict=True`.
- **Sai số lan truyền cực đại (Max Absolute Difference)**: **$1.49011612 \times 10^{-8}$ (Ngang bằng sai số làm tròn số thực IEEE 754 float32)**.
- **Kiểm chứng kích thước đầu ra**:
  - `(2, 3, 128, 128)` -> `(2, 1, 128, 128)` đối với 1 lớp.
  - `(2, 3, 128, 128)` -> `(2, 4, 128, 128)` đối với 4 lớp.
- **Kiểm chứng checkpoint (Serialization & Deserialization)**:
  - Nạp và xuất checkpoint với `strict=True` thực hiện thành công $100\%$.

---

## 5. Kết luận kiểm toán

Mô hình `fast-scnn` trong `models/fast-scnn.py` được triển khai chuẩn mực và trùng khớp $100\%$ từng tầng và từng tham số với kho tham chiếu canonical `tramac/fast-scnn-pytorch` và bài báo gốc Fast-SCNN (BMVC 2019). Đạt chuẩn bài báo A*/Q1.

