# BÁO CÁO KIỂM KÊ DANH MỤC MÔ HÌNH VÀ MÔI TRƯỜNG THỰC NGHIỆM (AUDIT INVENTORY)

**Dự án**: `FloodEdgeNet` (Phân vùng ảnh ngập lụt phục vụ triển khai thiết bị biên & vi điều khiển MCU)  
**Ngày thực hiện**: 01/11/2026  
**Vai trò**: Trưởng ban điều phối kiểm toán mã nguồn (Lead Orchestrator)  
**Tiêu chuẩn chất lượng**: Tiêu chuẩn tái lập phục vụ bài báo Q1 / A* (Zero tolerance for architectural deviation)

---

## 1. Môi trường thực thi & Giao thức thực nghiệm (Environment & Protocol)

Toàn bộ thông tin được trích xuất trực tiếp từ mã nguồn thực tế của dự án `FloodEdgeNet`:

- **Hệ điều hành**: Windows 11 (build 26100)
- **Python**: `3.14.0` (C:\Python314\python.exe)
- **PyTorch**: `2.14.0+cpu`
- **TorchVision**: `0.29.0+cpu`
- **timm**: `1.0.30`
- **segmentation_models_pytorch (smp)**: Không cài đặt trên môi trường máy chủ cục bộ (toàn bộ 11 mô hình baseline được triển khai độc lập dạng mã nguồn thuần PyTorch trong thư mục `models/`).
- **Thư viện đo lường độ phức tạp (Profiling)**:
  - `thop`: `0.1.1`
  - `ptflops`: `0.7.5`
  - `fvcore`: `0.1.5.post20221221`
- **Độ phân giải đầu vào (Input Resolution)**:
  - $128 \times 128 \times 3$ (mặc định tại `main.py` và `qat.py` để tối ưu cho thiết bị vi điều khiển MCU).
  - $256 \times 256 \times 3$ (được hỗ trợ và đánh giá tại `benchmark.py` và `benchmark_floodscene.py`).
- **Số lượng lớp đầu ra (Number of Classes)**:
  - Phân vùng nhị phân (Binary Flood Segmentation): `num_classes = 1` (áp dụng cho `floodkaggle`, `floodvn`, và biến thể nhị phân `aifloodsense`).
  - Phân vùng đa lớp (Multi-class Segmentation): `num_classes = 4` (áp dụng cho tập dữ liệu `aifloodsense` / `floodscene` gồm 4 lớp: 0: Background, 1: Sky, 2: Building, 3: Flood).
- **Chính sách tiền xử lý & Chuẩn hóa (Normalization Policy)**:
  - Tỷ lệ co dãn điểm ảnh: `image.float() / 255.0` đưa về dải $[0.0, 1.0]$ trong `utils/dataloader.py`.
  - Không áp dụng phép trừ Mean/Std ImageNet tại DataLoader; hình ảnh đầu vào được giữ nguyên chuẩn dải $[0, 1]$.
  - Thứ tự kênh màu: RGB, dữ liệu kiểu `torch.float32`.
- **Hàm mất mát (Loss Functions)**:
  - Nhị phân: `bce` (BCEWithLogitsLoss), `dice`, `focal`, `lovasz`, `bce_dice`.
  - Đa lớp: `ce` (CrossEntropyLoss), `ce_dice`, `focal`.
  - Mất mát phụ trợ (Auxiliary Loss): `boundary` (Boundary IoU Loss), `bce`, `dice` (trọng số `aux_weight = 0.4`).
- **Chỉ số đánh giá bài báo (Evaluation Metrics)**:
  - `mIoU` (Mean IoU), `Dice / F1`, `Pixel Accuracy`, `Precision`, `Recall`, `FWIoU` (Frequency Weighted IoU), `Boundary IoU`.
  - Chỉ số biên: `FPS`, `Latency (ms)`, `Peak SRAM (KB)`, `Flash Model Size (KB)`.
- **Tập dữ liệu & Phân chia (Datasets & Splits)**:
  - **FloodKaggle**: Phân chia phân tầng 70% Train, 15% Val, 15% Test với seed cố định `seed = 42`.
  - **AIFloodSense (FloodScene)**: Phân chia Train/Val/Test có sẵn theo thư mục hoặc phân chia ngẫu nhiên cố định 70/15/15.
  - **FloodVN**: Bộ dữ liệu ảnh lũ lụt thực tế tại Việt Nam (sử dụng nhãn polygon LabelMe JSON), được dùng làm chuẩn đánh giá chuyển giao miền (Zero-Shot Cross-Dataset Transfer Benchmark).
- **Hạt giống ngẫu nhiên (Seeds)**:
  - Seed mặc định: `42`.
  - Multi-seed đánh giá bài báo: `MULTI_SEEDS = [42, 123, 456, 789, 2024]`.

---

## 2. Danh mục 11 Mô hình Baseline So Sánh (In-Scope Baselines)

Dưới đây là 11 kiến trúc mô hình so sánh có mặt trong thư mục `models/` của dự án:

| STT | Tên Model | Đường dẫn tệp | Class thực thi | Nguồn gốc / Họ kiến trúc | Biến thể cụ thể (Variant) | Pretrained Tag | Output Stride | Tham số (1 lớp) |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| 1 | `dabnet` | `models/dabnet.py` | `DABNet` | BMVC 2019 (Li et al.) | Standard DABNet (DW-Asym-Bottleneck, dilation 2,4,8,16) | Không (Scratch init) | 8 | 751,981 |
| 2 | `deeplabv3` | `models/deeplabv3.py` | `DeepLabV3Model` | Google (Chen et al., 2017) / torchvision | ResNet-50 Dilated Backbone + ASPP (rates 6,12,18) | Hỗ trợ ImageNet (`--pretrained`) | 16 | 39,043,393 |
| 3 | `edanet` | `models/edanet.py` | `EDANetModel` | ICME 2019 (Lo et al.) | EDANet (Asymmetric 1D conv 3x1/1x3, dense, dilation 2,5,9) | Không (Scratch init) | 8 | 678,580 |
| 4 | `enet` | `models/enet.py` | `ENetModel` | CVPR 2016 workshop (Paszke et al.) | Standard ENet (5 stages, regular/dilated/asymmetric bottlenecks) | Không (Scratch init) | 8 | 365,950 |
| 5 | `espnetv2` | `models/espnetv2.py` | `ESPNetv2Segmentation` | CVPR 2019 (Mehta et al.) | EESPNet Backbone (stride 2/4/8/16/32, K=[1,2,4,8,16]) + PyrPool | Không (Scratch init) | 8 | 836,774 |
| 6 | `fast-scnn` | `models/fast-scnn.py` | `FastSCNN` | BMVC 2019 (Poudel et al.) | Fast-SCNN (Learning to Downsample, GFE, PPM, FFM) | Không (Scratch init) | 8 | 1,135,729 |
| 7 | `segformer` | `models/segformer.py` | `SegFormerModel` | NeurIPS 2021 (Xie et al.) / NVlabs | MiT-B0 Backbone (dims [32,64,160,256], depths [2,2,2,2]) + All-MLP Decoder | Hỗ trợ MiT-B0 (`--pretrained`) | 4 | 24,722,369 |
| 8 | `segnet` | `models/segnet.py` | `SegNetModel` | IEEE TPAMI 2017 (Badrinarayanan et al.) | SegNet (VGG-16 style encoder với Max Unpooling Indices) | Không (Scratch init) | 32 | 29,480,129 |
| 9 | `stdc1` | `models/stdc1.py` | `STDCModel` | CVPR 2021 (Fan et al.) | STDC1 Backbone ([32,64,256,512,1024]) + ARM + FFM + Detail Head | Không (Scratch init) | 8 | 7,786,144 |
| 10 | `ulite` | `models/ulite.py` | `ULiteModel` | IEEE TMI / arXiv 2023 | U-Lite (Axial Depthwise Conv 1x7 & 7x1, dims [16,32,64,128]) | Không (Scratch init) | 16 | 878,417 |
| 11 | `unet` | `models/unet.py` | `UNetModel` | MICCAI 2015 (Ronneberger et al.) | Standard 4-stage U-Net ([64,128,256,512,1024], ConvTranspose) | Không (Scratch init) | 16 | 31,043,521 |

---

## 3. Danh mục các mô hình nằm NGOÀI phạm vi kiểm toán (Out-of-Scope)

Theo chỉ thị nghiêm ngặt từ người dùng:
- **Mô hình đề xuất cốt lõi**: `flood_edgenet.py` (`FloodEdgeNet` / `fsenet`) - **TUYỆT ĐỐI KHÔNG AUDIT, KHÔNG SỬA ĐỔI**.
- **Bộ 23 mô hình nghiên cứu cắt bỏ thành phần (Ablation Suite)**:
  - `ablation2_ECA.py` đến `ablation24_AuxS1.py` - **TUYỆT ĐỐI KHÔNG AUDIT, KHÔNG SỬA ĐỔI**.

---

## 4. Kế hoạch triển khai Sub-Agent (Step 1 Phân bổ nhiệm vụ)

Mỗi mô hình trong số 11 baseline trên sẽ được phân bổ một quy trình thẩm định độc lập kiểm tra đầy đủ các mục:
- **A. Nguồn tham chiếu gốc**: Xác định đúng link repo GitHub chính thức của tác giả hoặc bài báo gốc, commit hash/release tag, DOI/arXiv.
- **B. Đối chiếu kiến trúc từng tầng**: Bảng diff chi tiết lớp-qua-lớp (kernel, stride, padding, dilation, groups, bias, BN, activation, pooling, skip connection, upsampling).
- **C. Xác minh tham số và độ phức tạp**: Đếm chính xác tham số và đo MACs/FLOPs tại $128 \times 128$ với `thop`, `ptflops`, và `fvcore`.
- **D. Kiểm chứng lan truyền xuôi (Forward-Pass)**: Nạp trọng số đối chuẩn, kiểm tra sai số logit cực đại $\le 10^{-5}$ và kích thước tensor các tầng.
- **E. Tính toàn vẹn trọng số và pipeline**: Kiểm tra checkpoint `load_state_dict(strict=True)`, kiểm tra tính đồng nhất giữa train và inference.
- **F. Vòng lặp sửa chữa và tái kiểm tra**: Tạo bản sao lưu `<file>.orig` nếu cần sửa bất kỳ lỗi nào, tái chạy toàn bộ kiểm tra cho tới khi đạt PASS.
- **G. Bàn giao**: Xuất bản `audit/<model>/verify.py` (KHÔNG comment, KHÔNG docstring) và `audit/<model>/report.md` (tiếng Việt).

