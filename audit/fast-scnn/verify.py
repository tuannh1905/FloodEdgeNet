import os
import sys
import importlib.util
import torch
from thop import profile as thop_profile
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

spec_ours = importlib.util.spec_from_file_location("ours_fastscnn", r"d:\AI Project\FloodEdgeNet\models\fast-scnn.py")
mod_ours = importlib.util.module_from_spec(spec_ours)
spec_ours.loader.exec_module(mod_ours)

spec_ref = importlib.util.spec_from_file_location("ref_fastscnn", r"d:\AI Project\FloodEdgeNet\audit\refs\fast-scnn\tramac\models\fast_scnn.py")
mod_ref = importlib.util.module_from_spec(spec_ref)
spec_ref.loader.exec_module(mod_ref)

torch.manual_seed(42)
img_size = 128
dummy_input = torch.randn(2, 3, img_size, img_size)

ref_19 = mod_ref.FastSCNN(19)
ref_1 = mod_ref.FastSCNN(1)
ref_4 = mod_ref.FastSCNN(4)

model_1 = mod_ours.build_model(num_classes=1, input_size=img_size)
model_4 = mod_ours.build_model(num_classes=4, input_size=img_size)

p_ref_19 = sum(p.numel() for p in ref_19.parameters())
p_ref_1 = sum(p.numel() for p in ref_1.parameters())
p_ref_4 = sum(p.numel() for p in ref_4.parameters())

p_1 = sum(p.numel() for p in model_1.parameters())
p_4 = sum(p.numel() for p in model_4.parameters())

assert p_1 == p_ref_1, f"Params mismatch 1 class: {p_1} vs {p_ref_1}"
assert p_4 == p_ref_4, f"Params mismatch 4 classes: {p_4} vs {p_ref_4}"

delta_1 = (128 * 1 + 1) - (128 * 19 + 19)
delta_4 = (128 * 4 + 4) - (128 * 19 + 19)
assert p_1 == p_ref_19 + delta_1, "Classifier delta mismatch for 1 class"
assert p_4 == p_ref_19 + delta_4, "Classifier delta mismatch for 4 classes"

ref_1.load_state_dict(model_1.state_dict(), strict=True)
ref_1.eval()
model_1.eval()

with torch.no_grad():
    out_ours = model_1(dummy_input)
    out_ref = ref_1(dummy_input)[0]

max_diff = torch.max(torch.abs(out_ours - out_ref)).item()
assert max_diff <= 1e-5, f"Forward diff exceeded: {max_diff}"
assert out_ours.shape == (2, 1, img_size, img_size), f"Output shape mismatch: {out_ours.shape}"

dummy_ckpt = {"model_state_dict": model_1.state_dict()}
ckpt_path = "temp_ckpt_fastscnn.pth"
torch.save(dummy_ckpt, ckpt_path)
loaded_ckpt = torch.load(ckpt_path, weights_only=False)
loaded_model = mod_ours.build_model(num_classes=1, input_size=img_size)
loaded_model.load_state_dict(loaded_ckpt["model_state_dict"], strict=True)
os.remove(ckpt_path)

model_prof = mod_ours.build_model(num_classes=1, input_size=img_size)
model_prof.eval()
thop_macs, thop_params = thop_profile(model_prof, inputs=(dummy_input[:1],), verbose=False)
pt_macs, pt_params = get_model_complexity_info(model_prof, (3, img_size, img_size), as_strings=False, print_per_layer_stat=False, verbose=False)
fv_flops = FlopCountAnalysis(model_prof, dummy_input[:1]).total()

print("Fast-SCNN Verification Summary:")
print(f"  Params Ref 19 (Cityscapes): {p_ref_19:,}")
print(f"  Params Ours 1 (Binary):     {p_1:,}")
print(f"  Params Ours 4 (Multi):      {p_4:,}")
print(f"  Max Logit Diff vs Ref:      {max_diff:.8e}")
print(f"  thop MACs:                  {thop_macs / 1e6:.2f} M")
print(f"  ptflops MACs:               {pt_macs / 1e6:.2f} M")
print(f"  fvcore FLOPs:               {fv_flops / 1e6:.2f} M")
print("  Strict Load:                SUCCESS")
print("  Status:                     PASS")

