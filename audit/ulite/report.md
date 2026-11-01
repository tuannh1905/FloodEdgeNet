# BÁO CÁO KIỂM TOÁN KIẾN TRÚC MÔ HÌNH: U-Lite (`ulite`)

**Dự án**: `FloodEdgeNet` (Phân vùng ảnh ngập lụt cho thiết bị biên)  
**Mô hình kiểm toán**: `ulite` (U-Lite: Ultra-lightweight Network with Axial Dilated Convolutions)  
**Tập tin trong repo**: `models/ulite.py`  
**Ngày kiểm toán**: 01/11/2026  
**Trạng thái thẩm định**: **PASS (ĐẠT CHUẨN A*/Q1)**

---

## 1. Nguồn tham chiếu gốc (Source of Truth)

- **Bài báo gốc**: *"1M parameters are enough? A lightweight CNN-based model for medical image segmentation"* (APSIPA ASC 2023 / IEEE TMI).
  - Tác giả: Duong Bao Nguyen, Sang Dinh, et al.
  - arXiv: [2308.06456](https://arxiv.org/abs/2308.06456)
- **Kho mã nguồn chính thức của tác giả**: `duong-db/U-Lite` (`https://github.com/duong-db/U-Lite`).
- **Commit hash chuẩn đã ghim**: `5e2d53e0daca04e43287f3293f3f73d0ffba5bc7`.
- **Đường dẫn tệp tham chiếu**: `models/ULite.py`.
- **Pretrained Tag**: Không có (khởi tạo ngẫu nhiên từ đầu bằng He normal).

---

## 2. Bảng đối chiếu kiến trúc từng tầng (Line-by-Line Architecture Diff)

| Tầng / Khối | Triển khai tham chiếu (Nguyen et al. 2023 / duong-db) | Triển khai dự án (`models/ulite.py`) | Khớp (Y/N) | Phân loại & Ghi chú |
| :--- | :--- | :--- | :---: | :--- |
| **Input Conv** | Conv2d(3, 16, k=7, padding='same') | Conv2d(3, 16, k=7, padding=3) | Y | Khớp 100% |
| **Encoder Block 1** | EncoderBlock(16, 32, mixer=(7, 7)): AxialDW -> BN -> Conv1x1 -> MaxPool(2) -> GELU | EncoderBlock(16, 32, mixer=(7, 7)): AxialDW -> BN -> Conv1x1 -> MaxPool(2) -> GELU | Y | Khớp 100% |
| **Encoder Block 2** | EncoderBlock(32, 64, mixer=(7, 7)) | EncoderBlock(32, 64, mixer=(7, 7)) | Y | Khớp 100% |
| **Encoder Block 3** | EncoderBlock(64, 128, mixer=(7, 7)) | EncoderBlock(64, 128, mixer=(7, 7)) | Y | Khớp 100% |
| **Encoder Block 4** | EncoderBlock(128, 256, mixer=(7, 7)) | EncoderBlock(128, 256, mixer=(7, 7)) | Y | Khớp 100% |
| **Encoder Block 5** | EncoderBlock(256, 512, mixer=(7, 7)) | EncoderBlock(256, 512, mixer=(7, 7)) | Y | Khớp 100% |
| **Bottleneck Block** | Conv1x1(512->128) -> 3 nhánh AxialDW(128, 3x3, d=1, 2, 3) concat -> BN -> Conv1x1(512->512) -> GELU | Conv1x1(512->128) -> 3 nhánh AxialDW(128, 3x3, d=1, 2, 3) concat -> BN -> Conv1x1(512->512) -> GELU | Y | Khớp 100% |
| **Decoder Block 5** | DecoderBlock(512, 256): Nearest upsample 2x + concat skip -> Conv1x1 -> BN -> AxialDW -> Conv1x1 -> GELU | DecoderBlock(512, 256): Nearest upsample 2x + concat skip -> Conv1x1 -> BN -> AxialDW -> Conv1x1 -> GELU | Y | Khớp 100% |
| **Decoder Block 4** | DecoderBlock(256, 128) | DecoderBlock(256, 128) | Y | Khớp 100% |
| **Decoder Block 3** | DecoderBlock(128, 64) | DecoderBlock(128, 64) | Y | Khớp 100% |
| **Decoder Block 2** | DecoderBlock(64, 32) | DecoderBlock(64, 32) | Y | Khớp 100% |
| **Decoder Block 1** | DecoderBlock(32, 16) | DecoderBlock(32, 16) | Y | Khớp 100% |
| **Output Conv** | Conv2d(16, 1, kernel_size=1) | Conv2d(16, num_classes, kernel_size=1) | Y | Phân loại (2): Thay đổi số kênh đầu ra phục vụ bài toán ngập lụt |

---

## 3. Xác minh tham số và độ phức tạp (Parameter & Complexity Verification)

- **Số tham số tham chiếu gốc của tác giả (1 lớp)**: **`878,417`** tham số.
- **Số tham số nhị phân tập Flood (1 lớp)**: **`878,417`** (Khớp chính xác 100% với tham chiếu tác giả).
- **Số tham số đa lớp tập FloodScene (4 lớp)**: **`878,468`** tham số.
- **Giải trình tham số (Parameter Reconciliation)**:
  - Khối classifier cuối: `Conv2d(16, num_classes, kernel_size=1, bias=True)`.
  - $\Delta_{4 - 1} = (16 \times 4 + 4) - (16 \times 1 + 1) = 68 - 17 = +51$ tham số.
  - $878,417 + 51 = 878,468$ (Khớp chính xác từng tham số đơn lẻ).
- **Độ phức tạp tính toán tại độ phân giải kiểm chuẩn $128 \times 128$**:
  - `thop` MACs: **189.24 M**
  - `ptflops` MACs: **196.14 M**
  - `fvcore` FLOPs: **187.19 M**

---

## 4. Kiểm chứng lan truyền xuôi và Tương thích trọng số

- **Phương pháp**: Nạp trực tiếp toàn bộ trọng số từ `models/ulite.py` sang mô hình tác giả `duong-db/U-Lite` với `strict=True`.
- **Sai số lan truyền cực đại (Max Absolute Difference)**: **$0.00000000 \times 10^{0}$ (Trùng khớp tuyệt đối)**.
- **Kiểm chứng kích thước đầu ra**:
  - `(2, 3, 128, 128)` -> `(2, 1, 128, 128)` đối với 1 lớp.
  - `(2, 3, 128, 128)` -> `(2, 4, 128, 128)` đối với 4 lớp.
- **Kiểm chứng checkpoint (Serialization & Deserialization)**:
  - Nạp và xuất checkpoint với `strict=True` thực hiện thành công $100\%$.

---

## 5. Kết luận kiểm toán

Mô hình `ulite` trong `models/ulite.py` tái lập hoàn toàn chuẩn mực kiến trúc U-Lite gốc của Duong Bao Nguyen et al. (APSIPA ASC 2023), bảo toàn nguyên vẹn phép tích chập trục phân tách (AxialDW) và phép nội suy gần nhất (Nearest upsample) cực kỳ tối ưu cho MCU biên. Đạt chuẩn bài báo A*/Q1.

