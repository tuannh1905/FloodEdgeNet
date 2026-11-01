import os
import sys
import torch
from thop import profile as thop_profile
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

sys.path.insert(0, r"d:\AI Project\FloodEdgeNet")

from models.enet import build_model, ENetModel

torch.manual_seed(42)
img_size = 128
dummy_input = torch.randn(2, 3, img_size, img_size)

model_1 = build_model(num_classes=1, input_size=img_size)
model_4 = build_model(num_classes=4, input_size=img_size)

p_1 = sum(p.numel() for p in model_1.parameters())
p_4 = sum(p.numel() for p in model_4.parameters())

delta_expected = (16 * 4 * 3 * 3) - (16 * 1 * 3 * 3)
assert p_4 - p_1 == delta_expected, f"Delta mismatch: {p_4 - p_1} vs {delta_expected}"
assert p_1 == 365950, f"Params 1 class mismatch: {p_1} vs 365950"
assert p_4 == 366382, f"Params 4 classes mismatch: {p_4} vs 366382"

model_1.eval()
model_4.eval()

with torch.no_grad():
    out_1 = model_1(dummy_input)
    out_4 = model_4(dummy_input)

assert out_1.shape == (2, 1, img_size, img_size), f"Output shape mismatch: {out_1.shape}"
assert out_4.shape == (2, 4, img_size, img_size), f"Output shape mismatch: {out_4.shape}"

dummy_ckpt = {"model_state_dict": model_1.state_dict()}
ckpt_path = "temp_ckpt_enet.pth"
torch.save(dummy_ckpt, ckpt_path)
loaded_ckpt = torch.load(ckpt_path, weights_only=False)
loaded_model = build_model(num_classes=1, input_size=img_size)
loaded_model.load_state_dict(loaded_ckpt["model_state_dict"], strict=True)
os.remove(ckpt_path)

model_prof = build_model(num_classes=1, input_size=img_size)
model_prof.eval()
thop_macs, thop_params = thop_profile(model_prof, inputs=(dummy_input[:1],), verbose=False)
pt_macs, pt_params = get_model_complexity_info(model_prof, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False, verbose=False)

fv_flops_str = "N/A (PyTorch JIT tracing bug in F.max_unpool2d tensor condition)"
try:
    fv_flops = FlopCountAnalysis(model_prof, dummy_input[:1]).total()
    fv_flops_str = f"{fv_flops / 1e6:.2f} M"
except Exception:
    pass

print("ENet Verification Summary:")
print(f"  Params Ours 1 (Binary):     {p_1:,}")
print(f"  Params Ours 4 (Multi):      {p_4:,}")
print(f"  thop MACs:                  {thop_macs / 1e6:.2f} M")
print(f"  ptflops MACs:               {pt_macs / 1e6:.2f} M")
print(f"  fvcore FLOPs:               {fv_flops_str}")
print("  Strict Load:                SUCCESS")
print("  Status:                     PASS")

