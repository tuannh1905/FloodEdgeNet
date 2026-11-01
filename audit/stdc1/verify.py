import os
import sys
import torch
from thop import profile as thop_profile
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

sys.path.insert(0, r"d:\AI Project\FloodEdgeNet")

from models.stdc1 import build_model, STDCModel

torch.manual_seed(42)
img_size = 128
dummy_input = torch.randn(2, 3, img_size, img_size)

model_1 = build_model(num_classes=1, encoder_type="stdc1", input_size=img_size)
model_4 = build_model(num_classes=4, encoder_type="stdc1", input_size=img_size)

p_1 = sum(p.numel() for p in model_1.parameters())
p_4 = sum(p.numel() for p in model_4.parameters())

delta_expected = 128 * 4 - 128 * 1
assert p_4 - p_1 == delta_expected, f"Delta mismatch: {p_4 - p_1} vs {delta_expected}"
assert p_1 == 7786144, f"Params 1 class mismatch: {p_1} vs 7786144"
assert p_4 == 7786528, f"Params 4 classes mismatch: {p_4} vs 7786528"

model_1.eval()
model_4.eval()

with torch.no_grad():
    out_1 = model_1(dummy_input)
    out_4 = model_4(dummy_input)

assert out_1.shape == (2, 1, img_size, img_size), f"Output shape mismatch: {out_1.shape}"
assert out_4.shape == (2, 4, img_size, img_size), f"Output shape mismatch: {out_4.shape}"

dummy_ckpt = {"model_state_dict": model_1.state_dict()}
ckpt_path = "temp_ckpt_stdc1.pth"
torch.save(dummy_ckpt, ckpt_path)
loaded_ckpt = torch.load(ckpt_path, weights_only=False)
loaded_model = build_model(num_classes=1, encoder_type="stdc1", input_size=img_size)
loaded_model.load_state_dict(loaded_ckpt["model_state_dict"], strict=True)
os.remove(ckpt_path)

model_prof = build_model(num_classes=1, encoder_type="stdc1", input_size=img_size)
model_prof.eval()
thop_macs, thop_params = thop_profile(model_prof, inputs=(dummy_input[:1],), verbose=False)
pt_macs, pt_params = get_model_complexity_info(model_prof, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False, verbose=False)
fv_flops = FlopCountAnalysis(model_prof, dummy_input[:1]).total()

print("STDC1 Verification Summary:")
print(f"  Params Ours 1 (Binary):     {p_1:,}")
print(f"  Params Ours 4 (Multi):      {p_4:,}")
print(f"  thop MACs:                  {thop_macs / 1e6:.2f} M")
print(f"  ptflops MACs:               {pt_macs / 1e6:.2f} M")
print(f"  fvcore FLOPs:               {fv_flops / 1e6:.2f} M")
print("  Strict Load:                SUCCESS")
print("  Status:                     PASS")

