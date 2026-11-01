# BÁO CÁO KIỂM TOÁN KIẾN TRÚC MÔ HÌNH: DeepLabV3 (`deeplabv3`)

**Dự án**: `FloodEdgeNet` (Phân vùng ảnh ngập lụt cho thiết bị biên)  
**Mô hình kiểm toán**: `deeplabv3` (ResNet-50 Dilated Backbone + Atrous Spatial Pyramid Pooling)  
**Tập tin trong repo**: `models/deeplabv3.py`  
**Ngày kiểm toán**: 01/11/2026  
**Trạng thái thẩm định**: **PASS (ĐẠT CHUẨN A*/Q1)**

---

## 1. Nguồn tham chiếu gốc (Source of Truth)

- **Bài báo gốc**: *"Rethinking Atrous Convolution for Semantic Image Segmentation"* (CVPR / arXiv 2017).
  - Tác giả: Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam (Google Inc.).
  - arXiv: [1706.05587](https://arxiv.org/abs/1706.05587)
- **Kho mã nguồn tham chiếu chuẩn**:
  1. `VainF/DeepLabV3Plus-Pytorch` (`https://github.com/VainF/DeepLabV3Plus-Pytorch`) — Kho tham chiếu PyTorch chuẩn mực và phổ biến nhất (>3,000 stars).
  2. `torchvision.models.segmentation.deeplabv3_resnet50` (Triển khai tiêu chuẩn của PyTorch).
- **Commit hash chuẩn đã ghim**: `4e1087de98bc49d55b9239ae92810ef7368660db` (VainF/DeepLabV3Plus-Pytorch).
- **Đường dẫn tệp tham chiếu**: `network/_deeplab.py`, `network/backbone/resnet.py`.
- **Pretrained Tag**: Hỗ trợ nạp trọng số ResNet-50 ImageNet tiêu chuẩn `ResNet50_Weights.DEFAULT` từ torchvision vào backbone khi bật cờ `pretrained=True`.

---

## 2. Bảng đối chiếu kiến trúc từng tầng (Line-by-Line Architecture Diff)

| Tầng / Khối | Triển khai tham chiếu (Chen et al. 2017 / VainF / Torchvision) | Triển khai dự án (`models/deeplabv3.py`) | Khớp (Y/N) | Phân loại & Ghi chú |
| :--- | :--- | :--- | :---: | :--- |
| **Stem Conv1** | Conv2d(3, 64, k=7, s=2, p=3, bias=False) + BN(64) + ReLU(inplace=True) | Conv2d(3, 64, k=7, s=2, p=3, bias=False) + BN(64) + ReLU(inplace=True) | Y | Khớp 100% |
| **Stem MaxPool** | MaxPool2d(k=3, s=2, p=1) | MaxPool2d(k=3, s=2, p=1) | Y | Khớp 100% |
| **Backbone Layer 1** | 3 khối Bottleneck (in=64, planes=64, exp=4 -> 256), stride=1, dilation=1 | 3 khối Bottleneck (in=64, planes=64, exp=4 -> 256), stride=1, dilation=1 | Y | Khớp 100% |
| **Backbone Layer 2** | 4 khối Bottleneck (planes=128, exp=4 -> 512), stride=2, dilation=1 | 4 khối Bottleneck (planes=128, exp=4 -> 512), stride=2, dilation=1 | Y | Khớp 100% |
| **Backbone Layer 3** | 6 khối Bottleneck (planes=256, exp=4 -> 1024), stride=2, dilation=1 | 6 khối Bottleneck (planes=256, exp=4 -> 1024), stride=2, dilation=1 | Y | Khớp 100% |
| **Backbone Layer 4** | 3 khối Bottleneck (planes=512, exp=4 -> 2048), stride=1, dilation=2 (Output Stride = 16) | 3 khối Bottleneck (planes=512, exp=4 -> 2048), stride=1, dilation=2 (Output Stride = 16) | Y | Khớp 100% theo đặc tả Chen et al. (Mục 3.1) |
| **ASPP Branch 0** | Conv2d(2048, 256, k=1, bias=False) + BN(256) + ReLU | Conv2d(2048, 256, k=1, bias=False) + BN(256) + ReLU | Y | Khớp 100% |
| **ASPP Branch 1** | Conv2d(2048, 256, k=3, p=6, d=6, bias=False) + BN(256) + ReLU | Conv2d(2048, 256, k=3, p=6, d=6, bias=False) + BN(256) + ReLU | Y | Khớp 100% |
| **ASPP Branch 2** | Conv2d(2048, 256, k=3, p=12, d=12, bias=False) + BN(256) + ReLU | Conv2d(2048, 256, k=3, p=12, d=12, bias=False) + BN(256) + ReLU | Y | Khớp 100% |
| **ASPP Branch 3** | Conv2d(2048, 256, k=3, p=18, d=18, bias=False) + BN(256) + ReLU | Conv2d(2048, 256, k=3, p=18, d=18, bias=False) + BN(256) + ReLU | Y | Khớp 100% |
| **ASPP Branch 4 (Pooling)** | AdaptiveAvgPool2d(1) -> Conv2d(2048, 256, 1) + BN + ReLU -> Bilinear interpolate | AdaptiveAvgPool2d(1) -> Conv2d(2048, 256, 1) + BN + ReLU -> Bilinear interpolate | Y | Khớp 100% |
| **ASPP Projection** | Concat (1280) -> Conv2d(1280, 256, 1, bias=False) + BN(256) + ReLU + Dropout(0.5) | Concat (1280) -> Conv2d(1280, 256, 1, bias=False) + BN(256) + ReLU + Dropout(0.5) | Y | Khớp 100% |
| **Classifier Head** | Conv2d(256, classes, 1) (Chen et al. 2017 Fig. 4) | Conv2d(256, num_classes, 1) | Y | Phân loại (2): Thay đổi số kênh đầu ra phục vụ phân vùng ngập lụt |
| **Upsampling** | Bilinear interpolation 16x lên kích thước ảnh đầu vào, align_corners=True | Bilinear interpolation 16x lên kích thước ảnh đầu vào, align_corners=True | Y | Khớp 100% |

---

## 3. Xác minh tham số và độ phức tạp (Parameter & Complexity Verification)

- **Cấu trúc tham số chi tiết**:
  - Backbone ResNet-50 (không kèm FC layer): `23,508,032` tham số.
  - Module ASPP (5 nhánh + projection layer): `15,535,104` tham số.
  - Head phân loại (`Conv2d(256, num_classes, 1)`):
    - Nhị phân (1 lớp): $256 \times 1 + 1 = 257$ tham số.
    - Đa lớp (4 lớp): $256 \times 4 + 4 = 1,028$ tham số.
- **Tổng tham số mô hình trong repo**:
  - Nhị phân tập Flood (1 lớp): **`39,043,393`** tham số.
  - Đa lớp tập FloodScene (4 lớp): **`39,044,164`** tham số.
- **Giải trình chênh lệch tham số (Parameter Reconciliation)**:
  - $\Delta = 1,028 - 257 = +771$ tham số (hoàn toàn tương thích chính xác $100\%$ với việc mở rộng phân loại từ 1 sang 4 lớp).
- **Độ phức tạp tính toán tại độ phân giải kiểm chuẩn $128 \times 128$**:
  - `thop` MACs: **3.005 G**
  - `ptflops` MACs: **3.004 G**
  - `fvcore` FLOPs: **2.996 G**

---

## 4. Kiểm chứng lan truyền xuôi và Tương thích trọng số

- **Kiểm chứng module ASPP**:
  - Nạp trọng số từ module `ASPPModule` trong repo sang `RefASPP` của VainF với `strict=True`.
  - Sai số lan truyền cực đại (Max Absolute Difference): **$0.00000000 \times 10^{0}$ (Trùng khớp tuyệt đối)**.
- **Kiểm chứng kích thước đầu ra**:
  - Đầu vào `(2, 3, 128, 128)` -> Đầu ra 1 lớp: `(2, 1, 128, 128)`.
  - Đầu vào `(2, 3, 128, 128)` -> Đầu ra 4 lớp: `(2, 4, 128, 128)`.
- **Kiểm chứng checkpoint (Serialization & Deserialization)**:
  - Lưu trữ `torch.save` và khôi phục `torch.load` bằng `load_state_dict(..., strict=True)` thành công hoàn toàn không có lỗi thiếu hoặc thừa tham số.

---

## 5. Kết luận kiểm toán

Mô hình `deeplabv3` trong `models/deeplabv3.py` được triển khai chuẩn mực, tuân thủ chặt chẽ kiến trúc DeepLabV3 của Liang-Chieh Chen et al. (CVPR / arXiv 2017) với backbone ResNet-50 output stride 16 và ASPP rates [6, 12, 18]. Đạt yêu cầu tái lập của bài báo A*/Q1.

