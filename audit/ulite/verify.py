import os
import sys
import importlib.util
import torch
from thop import profile as thop_profile
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

spec_ref = importlib.util.spec_from_file_location("ref_ulite", r"d:\AI Project\FloodEdgeNet\audit\refs\ulite\models\ULite.py")
mod_ref = importlib.util.module_from_spec(spec_ref)
spec_ref.loader.exec_module(mod_ref)

sys.path.insert(0, r"d:\AI Project\FloodEdgeNet")

from models.ulite import build_model, ULiteModel

torch.manual_seed(42)
img_size = 128
dummy_input = torch.randn(2, 3, img_size, img_size)

ref_1 = mod_ref.ULite()
model_1 = build_model(num_classes=1, input_size=img_size)
model_4 = build_model(num_classes=4, input_size=img_size)

p_ref = sum(p.numel() for p in ref_1.parameters())
p_1 = sum(p.numel() for p in model_1.parameters())
p_4 = sum(p.numel() for p in model_4.parameters())

delta_expected = (16 * 4 + 4) - (16 * 1 + 1)
assert p_4 - p_1 == delta_expected, f"Delta mismatch: {p_4 - p_1} vs {delta_expected}"
assert p_1 == p_ref, f"Params mismatch with ref: {p_1} vs {p_ref}"
assert p_1 == 878417, f"Params 1 class mismatch: {p_1} vs 878417"
assert p_4 == 878468, f"Params 4 classes mismatch: {p_4} vs 878468"

ref_1.load_state_dict(model_1.state_dict(), strict=True)
ref_1.eval()
model_1.eval()

with torch.no_grad():
    out_ours = model_1(dummy_input)
    out_ref = ref_1(dummy_input)

max_diff = torch.max(torch.abs(out_ours - out_ref)).item()
assert max_diff <= 1e-5, f"Forward diff exceeded: {max_diff}"
assert out_ours.shape == (2, 1, img_size, img_size), f"Output shape mismatch: {out_ours.shape}"

dummy_ckpt = {"model_state_dict": model_1.state_dict()}
ckpt_path = "temp_ckpt_ulite.pth"
torch.save(dummy_ckpt, ckpt_path)
loaded_ckpt = torch.load(ckpt_path, weights_only=False)
loaded_model = build_model(num_classes=1, input_size=img_size)
loaded_model.load_state_dict(loaded_ckpt["model_state_dict"], strict=True)
os.remove(ckpt_path)

model_prof = build_model(num_classes=1, input_size=img_size)
model_prof.eval()
thop_macs, thop_params = thop_profile(model_prof, inputs=(dummy_input[:1],), verbose=False)
pt_macs, pt_params = get_model_complexity_info(model_prof, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False, verbose=False)
fv_flops = FlopCountAnalysis(model_prof, dummy_input[:1]).total()

print("U-Lite Verification Summary:")
print(f"  Params Ref 1 (Official):    {p_ref:,}")
print(f"  Params Ours 1 (Binary):     {p_1:,}")
print(f"  Params Ours 4 (Multi):      {p_4:,}")
print(f"  Max Logit Diff vs Ref:      {max_diff:.8e}")
print(f"  thop MACs:                  {thop_macs / 1e6:.2f} M")
print(f"  ptflops MACs:               {pt_macs / 1e6:.2f} M")
print(f"  fvcore FLOPs:               {fv_flops / 1e6:.2f} M")
print("  Strict Load:                SUCCESS")
print("  Status:                     PASS")

