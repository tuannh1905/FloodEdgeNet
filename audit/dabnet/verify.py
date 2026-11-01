import os
import sys
import torch
from thop import profile as thop_profile
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

sys.path.insert(0, r"d:\AI Project\FloodEdgeNet")
sys.path.insert(0, r"d:\AI Project\FloodEdgeNet\audit\refs\dabnet")

from models.dabnet import DABNet as OursDABNet, build_model
from model.DABNet import DABNet as RefDABNet

torch.manual_seed(42)
img_size = 128
dummy_input = torch.randn(2, 3, img_size, img_size)

ref_19 = RefDABNet(classes=19)
ref_1 = RefDABNet(classes=1)
ref_4 = RefDABNet(classes=4)

ours_1 = build_model(num_classes=1, input_size=img_size)
ours_4 = build_model(num_classes=4, input_size=img_size)

p_ref_19 = sum(p.numel() for p in ref_19.parameters())
p_ref_1 = sum(p.numel() for p in ref_1.parameters())
p_ref_4 = sum(p.numel() for p in ref_4.parameters())

p_ours_1 = sum(p.numel() for p in ours_1.parameters())
p_ours_4 = sum(p.numel() for p in ours_4.parameters())

assert p_ref_1 == p_ours_1, f"Params mismatch 1 class: {p_ref_1} vs {p_ours_1}"
assert p_ref_4 == p_ours_4, f"Params mismatch 4 classes: {p_ref_4} vs {p_ours_4}"

delta_1 = (259 * 1) - (259 * 19)
delta_4 = (259 * 4) - (259 * 19)
assert p_ours_1 == p_ref_19 + delta_1, "Classifier delta mismatch for 1 class"
assert p_ours_4 == p_ref_19 + delta_4, "Classifier delta mismatch for 4 classes"

ours_1.load_state_dict(ref_1.state_dict(), strict=True)
ours_1.eval()
ref_1.eval()

with torch.no_grad():
    out_ours = ours_1(dummy_input)
    out_ref = ref_1(dummy_input)

max_diff = torch.max(torch.abs(out_ours - out_ref)).item()
assert max_diff <= 1e-5, f"Forward diff exceeded: {max_diff}"
assert out_ours.shape == (2, 1, img_size, img_size), f"Output shape mismatch: {out_ours.shape}"

dummy_ckpt = {"model_state_dict": ours_1.state_dict()}
ckpt_path = "temp_ckpt_dabnet.pth"
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

print("DABNet Verification Summary:")
print(f"  Params Ref 19 (Cityscapes): {p_ref_19:,}")
print(f"  Params Ours 1 (Binary):     {p_ours_1:,}")
print(f"  Params Ours 4 (Multi):      {p_ours_4:,}")
print(f"  thop MACs:                  {thop_macs / 1e6:.2f} M")
print(f"  ptflops MACs:               {pt_macs / 1e6:.2f} M")
print(f"  fvcore FLOPs:               {fv_flops / 1e6:.2f} M")
print(f"  Max Logit Diff:             {max_diff:.8e}")
print("  Strict Load:                SUCCESS")
print("  Status:                     PASS")

