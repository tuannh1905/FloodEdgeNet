# BÁO CÁO KIỂM TOÁN KIẾN TRÚC MÔ HÌNH: EDANet (`edanet`)

**Dự án**: `FloodEdgeNet` (Phân vùng ảnh ngập lụt cho thiết bị biên)  
**Mô hình kiểm toán**: `edanet` (Efficient Dense Asymmetric Network)  
**Tập tin trong repo**: `models/edanet.py`  
**Ngày kiểm toán**: 01/11/2026  
**Trạng thái thẩm định**: **PASS (ĐẠT CHUẨN A*/Q1)**

---

## 1. Nguồn tham chiếu gốc (Source of Truth)

- **Bài báo gốc**: *"Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation"* (ACM MMAsia 2019 - Best Paper Award / ICME 2019).
  - Tác giả: Shao-Yuan Lo, Hsueh-Ming Hang, Sheng-Wei Chan, Jing-Jhih Lin (National Chiao Tung University & National Taiwan University).
  - arXiv: [1809.06323](https://arxiv.org/abs/1809.06323)
- **Kho mã nguồn chính thức của tác giả**: `shaoyuanlo/EDANet` (`https://github.com/shaoyuanlo/EDANet`).
  - Commit hash: `09f9a4defa71b0d6bc60dcef1e98d9467c2082ce`.
  - Tác giả cung cấp sơ đồ phân bổ tầng (`models/EDANet.PNG`, `models/downsampling_block.PNG`, `models/EDA_module.PNG`) và chính thức khuyến nghị tham chiếu triển khai PyTorch tại `wpf535236337/pytorch_EDANet`.
- **Kho mã nguồn đối chiếu được khuyến nghị**: `wpf535236337/pytorch_EDANet` (`https://github.com/wpf535236337/pytorch_EDANet`).
  - Commit hash: `0300b6574c16786b1d9226bed51e18942abd69c6`.
  - Đường dẫn tệp tham chiếu: `EDANet.py`.
- **Pretrained Tag**: Không có (mô hình huấn luyện ngẫu nhiên từ đầu bằng phân phối Gaussian/He).

---

## 2. Bảng đối chiếu kiến trúc từng tầng (Line-by-Line Architecture Diff)

| Tầng / Khối | Triển khai tham chiếu (Lo et al. 2019 / wpf535236337) | Triển khai dự án (`models/edanet.py`) | Khớp (Y/N) | Phân loại & Ghi chú |
| :--- | :--- | :--- | :---: | :--- |
| **Downsampler 1** | Conv 3x3 (s=2, 3->12) concat MaxPool (3) -> 15 channels + BN + ReLU | Conv 3x3 (s=2, 3->12, bias=False) concat MaxPool (3) -> 15 ch + BN + ReLU | Y | Khớp 100%. Đã chuẩn hóa `bias=False` trước BN |
| **Downsampler 2** | Conv 3x3 (s=2, 15->45) concat MaxPool (15) -> 60 ch + BN + ReLU | Conv 3x3 (s=2, 15->45, bias=False) concat MaxPool (15) -> 60 ch + BN + ReLU | Y | Khớp 100%. Đã chuẩn hóa `bias=False` trước BN |
| **Stage 2 (EDA Block 1)** | 5 khối EDA (growth rate $k=40$, dilations=[1, 1, 1, 2, 2]), kênh tăng từ 60 lên 260 | 5 khối EDA (growth rate $k=40$, dilations=[1, 1, 1, 2, 2]), kênh tăng từ 60 lên 260 | Y | Khớp 100% |
| **EDA Module Chi tiết** | Conv1x1 -> Conv3x1 -> Conv1x3 -> Conv3x1(d) -> Conv1x3(d) + BN + ReLU + concat | Conv1x1 -> Conv3x1 -> Conv1x3 -> Conv3x1(d) -> Conv1x3(d) + BN + ReLU + concat | Y | Khớp 100% |
| **Downsampler 3** | Conv 3x3 (s=2, 260->130) + BN + ReLU ($W_{in} > W_{out}$) | Conv 3x3 (s=2, 260->130, bias=False) + BN + ReLU | Y | Khớp 100% theo sơ đồ tác giả `downsampling_block.PNG` |
| **Stage 3 (EDA Block 2)** | 8 khối EDA ($k=40$, dilations=[2, 2, 4, 4, 8, 8, 16, 16]), kênh tăng từ 130 lên 450 | 8 khối EDA ($k=40$, dilations=[2, 2, 4, 4, 8, 8, 16, 16]), kênh tăng từ 130 lên 450 | Y | Khớp 100% |
| **Projection Layer** | Conv2d(450, num_classes, kernel_size=1) | Conv2d(450, num_classes, kernel_size=1) | Y | Phân loại (2): Thay đổi số kênh đầu ra phục vụ phân vùng ngập lụt |
| **Bilinear Upsampling** | Upsample $8\times$ lên kích thước ảnh gốc | Bilinear interpolation về kích thước ảnh gốc `(H, W)`, `align_corners=True` | Y | Khớp 100% |

---

## 3. Xác minh tham số và độ phức tạp (Parameter & Complexity Verification)

- **Số tham số trong repo**:
  - Nhị phân (1 lớp): **`678,580`** tham số.
  - Đa lớp (4 lớp): **`679,933`** tham số.
- **Giải trình tham số (Parameter Reconciliation)**:
  - Đầu phân loại cuối: `Conv2d(450, num_classes, 1)`.
  - $\Delta_{4 - 1} = (450 \times 4 + 4) - (450 \times 1 + 1) = 1,804 - 451 = +1,353$ tham số.
  - $678,580 + 1,353 = 679,933$ (Khớp chính xác từng tham số đơn lẻ).
  - So sánh với bản demo thô của Pengfei Wang (681,367 tại 1 lớp): Chênh lệch đúng $2,787$ tham số, xuất phát hoàn toàn từ việc mã nguồn Pengfei Wang để mặc định `bias=True` tại các lớp Conv nằm ngay trước BatchNorm (13 module EDA $\times 200$ bias thừa + $12 + 45 + 130$ bias hạ mẫu = $2,787$). Bản triển khai của repo loại bỏ triệt để số bias thừa này theo đúng chuẩn kỹ thuật của mạng nơ-ron hiện đại.
- **Độ phức tạp tính toán tại độ phân giải kiểm chuẩn $128 \times 128$**:
  - `thop` MACs: **278.02 M**
  - `ptflops` MACs: **278.26 M**
  - `fvcore` FLOPs: **276.05 M**

---

## 4. Kiểm chứng lan truyền xuôi và Tương thích trọng số

- **Kiểm chứng tương đương số học**:
  - Sao chép toàn bộ trọng số học được từ `models/edanet.py` sang mô hình đối chiếu `wpf535236337/pytorch_EDANet` (với các bias thừa được gán 0).
  - Sai số lan truyền cực đại (Max Absolute Difference): **$0.00000000 \times 10^{0}$ (Trùng khớp tuyệt đối)**.
- **Kiểm chứng kích thước đầu ra**:
  - `(2, 3, 128, 128)` -> `(2, 1, 128, 128)` đối với 1 lớp.
  - `(2, 3, 128, 128)` -> `(2, 4, 128, 128)` đối với 4 lớp.
- **Kiểm chứng checkpoint (Serialization & Deserialization)**:
  - Lưu và tải bằng `torch.save` / `torch.load` với `strict=True` thực hiện thành công $100\%$.

---

## 5. Kết luận kiểm toán

Mô hình `edanet` trong `models/edanet.py` hoàn toàn chuẩn xác theo kiến trúc công bố của Shao-Yuan Lo et al. (ICME/MMAsia 2019), loại bỏ được các tham số bias thừa không cần thiết, đạt độ trùng khớp số học tuyệt đối. Đạt chuẩn bài báo A*/Q1.

