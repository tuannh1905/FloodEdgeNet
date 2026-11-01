import os
import sys
import torch
from thop import profile as thop_profile
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

sys.path.insert(0, r"d:\AI Project\FloodEdgeNet")
sys.path.insert(0, r"d:\AI Project\FloodEdgeNet\audit\refs\edanet\pytorch_EDANet")

from models.edanet import build_model, EDANetModel
from EDANet import EDANet as RefEDANet

torch.manual_seed(42)
img_size = 128
dummy_input = torch.randn(2, 3, img_size, img_size)

model_1 = build_model(num_classes=1, input_size=img_size)
model_4 = build_model(num_classes=4, input_size=img_size)

p_1 = sum(p.numel() for p in model_1.parameters())
p_4 = sum(p.numel() for p in model_4.parameters())

delta_expected = (450 * 4 + 4) - (450 * 1 + 1)
assert p_4 - p_1 == delta_expected, f"Delta mismatch: {p_4 - p_1} vs {delta_expected}"
assert p_1 == 678580, f"Params 1 class mismatch: {p_1} vs 678580"
assert p_4 == 679933, f"Params 4 classes mismatch: {p_4} vs 679933"

ref = RefEDANet(num_classes=1).eval()
for p in ref.parameters():
    if p.dim() == 1 and p.numel() not in [1, 4]:
        p.data.zero_()

ref.layers[0].conv.weight.data.copy_(model_1.stage1.conv.weight.data)
ref.layers[0].bn.weight.data.copy_(model_1.stage1.bn.weight.data)
ref.layers[0].bn.bias.data.copy_(model_1.stage1.bn.bias.data)
ref.layers[1].conv.weight.data.copy_(model_1.stage2_down.conv.weight.data)
ref.layers[1].bn.weight.data.copy_(model_1.stage2_down.bn.weight.data)
ref.layers[1].bn.bias.data.copy_(model_1.stage2_down.bn.bias.data)

for i in range(5):
    mod = model_1.stage2.modules_list[i]
    rmod = ref.layers[2 + i]
    rmod.conv1x1.weight.data.copy_(mod.conv1.weight.data)
    rmod.bn0.weight.data.copy_(mod.bn1.weight.data)
    rmod.bn0.bias.data.copy_(mod.bn1.bias.data)
    rmod.conv3x1_1.weight.data.copy_(mod.conv2.weight.data)
    rmod.conv1x3_1.weight.data.copy_(mod.conv3.weight.data)
    rmod.bn1.weight.data.copy_(mod.bn3.weight.data)
    rmod.bn1.bias.data.copy_(mod.bn3.bias.data)
    rmod.conv3x1_2.weight.data.copy_(mod.conv4.weight.data)
    rmod.conv1x3_2.weight.data.copy_(mod.conv5.weight.data)
    rmod.bn2.weight.data.copy_(mod.bn5.weight.data)
    rmod.bn2.bias.data.copy_(mod.bn5.bias.data)

ref.layers[7].conv.weight.data.copy_(model_1.stage3_down_conv.weight.data)
ref.layers[7].bn.weight.data.copy_(model_1.stage3_down_bn.weight.data)
ref.layers[7].bn.bias.data.copy_(model_1.stage3_down_bn.bias.data)

for j in range(8):
    mod = model_1.stage3.modules_list[j]
    rmod = ref.layers[8 + j]
    rmod.conv1x1.weight.data.copy_(mod.conv1.weight.data)
    rmod.bn0.weight.data.copy_(mod.bn1.weight.data)
    rmod.bn0.bias.data.copy_(mod.bn1.bias.data)
    rmod.conv3x1_1.weight.data.copy_(mod.conv2.weight.data)
    rmod.conv1x3_1.weight.data.copy_(mod.conv3.weight.data)
    rmod.bn1.weight.data.copy_(mod.bn3.weight.data)
    rmod.bn1.bias.data.copy_(mod.bn3.bias.data)
    rmod.conv3x1_2.weight.data.copy_(mod.conv4.weight.data)
    rmod.conv1x3_2.weight.data.copy_(mod.conv5.weight.data)
    rmod.bn2.weight.data.copy_(mod.bn5.weight.data)
    rmod.bn2.bias.data.copy_(mod.bn5.bias.data)

ref.project_layer.weight.data.copy_(model_1.project.weight.data)
ref.project_layer.bias.data.copy_(model_1.project.bias.data)

model_1.eval()
with torch.no_grad():
    out_ours = model_1(dummy_input)
    feat = dummy_input
    for layer in ref.layers:
        feat = layer(feat)
    out_ref_proj = ref.project_layer(feat)
    out_ref = torch.nn.functional.interpolate(out_ref_proj, size=(img_size, img_size), mode="bilinear", align_corners=True)

max_diff = torch.max(torch.abs(out_ours - out_ref)).item()
assert max_diff <= 1e-5, f"Forward diff exceeded: {max_diff}"
assert out_ours.shape == (2, 1, img_size, img_size), f"Output shape mismatch: {out_ours.shape}"

dummy_ckpt = {"model_state_dict": model_1.state_dict()}
ckpt_path = "temp_ckpt_edanet.pth"
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

print("EDANet Verification Summary:")
print(f"  Params Ours 1 (Binary):     {p_1:,}")
print(f"  Params Ours 4 (Multi):      {p_4:,}")
print(f"  Max Logit Diff vs Ref:      {max_diff:.8e}")
print(f"  thop MACs:                  {thop_macs / 1e6:.2f} M")
print(f"  ptflops MACs:               {pt_macs / 1e6:.2f} M")
print(f"  fvcore FLOPs:               {fv_flops / 1e6:.2f} M")
print("  Strict Load:                SUCCESS")
print("  Status:                     PASS")

