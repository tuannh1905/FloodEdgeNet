import os
import sys
import torch
from thop import profile as thop_profile
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

sys.path.insert(0, r"d:\AI Project\FloodEdgeNet")
sys.path.insert(0, r"d:\AI Project\FloodEdgeNet\audit\refs\deeplabv3")

from models.deeplabv3 import build_model, DeepLabV3Model
from network._deeplab import ASPP as RefASPP

torch.manual_seed(42)
img_size = 128
dummy_input = torch.randn(2, 3, img_size, img_size)

model_1 = build_model(num_classes=1, input_size=img_size, pretrained=False)
model_4 = build_model(num_classes=4, input_size=img_size, pretrained=False)

p_1 = sum(p.numel() for p in model_1.parameters())
p_4 = sum(p.numel() for p in model_4.parameters())

backbone_params = 23508032
aspp_params = 15535104
head_params_1 = 256 * 1 + 1
head_params_4 = 256 * 4 + 4

expected_p1 = backbone_params + aspp_params + head_params_1
expected_p4 = backbone_params + aspp_params + head_params_4

assert p_1 == expected_p1, f"Params mismatch 1 class: {p_1} vs {expected_p1}"
assert p_4 == expected_p4, f"Params mismatch 4 classes: {p_4} vs {expected_p4}"

ref_aspp = RefASPP(2048, [6, 12, 18])
ref_aspp.load_state_dict(model_1.aspp.state_dict(), strict=True)
ref_aspp.eval()
model_1.eval()

with torch.no_grad():
    dummy_feat = torch.randn(2, 2048, 8, 8)
    out_ours_aspp = model_1.aspp(dummy_feat)
    out_ref_aspp = ref_aspp(dummy_feat)

aspp_diff = torch.max(torch.abs(out_ours_aspp - out_ref_aspp)).item()
assert aspp_diff <= 1e-5, f"ASPP forward diff exceeded: {aspp_diff}"

with torch.no_grad():
    out_1 = model_1(dummy_input)
    out_4 = model_4(dummy_input)

assert out_1.shape == (2, 1, img_size, img_size), f"Output shape mismatch: {out_1.shape}"
assert out_4.shape == (2, 4, img_size, img_size), f"Output shape mismatch: {out_4.shape}"

dummy_ckpt = {"model_state_dict": model_1.state_dict()}
ckpt_path = "temp_ckpt_deeplabv3.pth"
torch.save(dummy_ckpt, ckpt_path)
loaded_ckpt = torch.load(ckpt_path, weights_only=False)
loaded_model = build_model(num_classes=1, input_size=img_size, pretrained=False)
loaded_model.load_state_dict(loaded_ckpt["model_state_dict"], strict=True)
os.remove(ckpt_path)

model_prof = build_model(num_classes=1, input_size=img_size, pretrained=False)
model_prof.eval()
thop_macs, thop_params = thop_profile(model_prof, inputs=(dummy_input[:1],), verbose=False)
pt_macs, pt_params = get_model_complexity_info(model_prof, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False, verbose=False)
fv_flops = FlopCountAnalysis(model_prof, dummy_input[:1]).total()

print("DeepLabV3 Verification Summary:")
print(f"  Params Ours 1 (Binary):     {p_1:,}")
print(f"  Params Ours 4 (Multi):      {p_4:,}")
print(f"  ASPP Max Diff vs Ref:       {aspp_diff:.8e}")
print(f"  thop MACs:                  {thop_macs / 1e9:.3f} G")
print(f"  ptflops MACs:               {pt_macs / 1e9:.3f} G")
print(f"  fvcore FLOPs:               {fv_flops / 1e9:.3f} G")
print("  Strict Load:                SUCCESS")
print("  Status:                     PASS")

