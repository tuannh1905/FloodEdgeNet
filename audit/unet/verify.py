import os
import sys
import torch
from thop import profile as thop_profile
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis

sys.path.insert(0, r"d:\AI Project\FloodEdgeNet\audit\refs\unet")
sys.path.insert(0, r"d:\AI Project\FloodEdgeNet")

from unet.unet_model import UNet as RefUNet
from models.unet import build_model, UNetModel

torch.manual_seed(42)
img_size = 128
dummy_input = torch.randn(2, 3, img_size, img_size)

ref_1 = RefUNet(3, 1, bilinear=False)
model_1 = build_model(num_classes=1, input_size=img_size)
model_4 = build_model(num_classes=4, input_size=img_size)

p_ref = sum(p.numel() for p in ref_1.parameters())
p_1 = sum(p.numel() for p in model_1.parameters())
p_4 = sum(p.numel() for p in model_4.parameters())

delta_expected = (64 * 4 + 4) - (64 * 1 + 1)
assert p_4 - p_1 == delta_expected, f"Delta mismatch: {p_4 - p_1} vs {delta_expected}"
assert p_1 == 31043521, f"Params 1 class mismatch: {p_1} vs 31043521"
assert p_4 == 31043716, f"Params 4 classes mismatch: {p_4} vs 31043716"
assert p_1 - p_ref == 5888, f"Bias reconciliation mismatch: {p_1 - p_ref} vs 5888"

ref_1.eval()
model_1.eval()

with torch.no_grad():
    for name, param in model_1.named_parameters():
        if "bias" in name and "bn" not in name and "outc" not in name:
            param.zero_()

    def copy_dc(src_dc, tgt_dc):
        tgt_dc.double_conv[0].weight.copy_(src_dc.conv1.weight)
        tgt_dc.double_conv[1].weight.copy_(src_dc.bn1.weight)
        tgt_dc.double_conv[1].bias.copy_(src_dc.bn1.bias)
        tgt_dc.double_conv[1].running_mean.copy_(src_dc.bn1.running_mean)
        tgt_dc.double_conv[1].running_var.copy_(src_dc.bn1.running_var)
        tgt_dc.double_conv[3].weight.copy_(src_dc.conv2.weight)
        tgt_dc.double_conv[4].weight.copy_(src_dc.bn2.weight)
        tgt_dc.double_conv[4].bias.copy_(src_dc.bn2.bias)
        tgt_dc.double_conv[4].running_mean.copy_(src_dc.bn2.running_mean)
        tgt_dc.double_conv[4].running_var.copy_(src_dc.bn2.running_var)

    copy_dc(model_1.inc, ref_1.inc)
    copy_dc(model_1.down1.conv, ref_1.down1.maxpool_conv[1])
    copy_dc(model_1.down2.conv, ref_1.down2.maxpool_conv[1])
    copy_dc(model_1.down3.conv, ref_1.down3.maxpool_conv[1])
    copy_dc(model_1.bottleneck.conv, ref_1.down4.maxpool_conv[1])
    ref_1.up1.up.weight.copy_(model_1.up1.up.weight)
    ref_1.up1.up.bias.copy_(model_1.up1.up.bias)
    copy_dc(model_1.up1.conv, ref_1.up1.conv)
    ref_1.up2.up.weight.copy_(model_1.up2.up.weight)
    ref_1.up2.up.bias.copy_(model_1.up2.up.bias)
    copy_dc(model_1.up2.conv, ref_1.up2.conv)
    ref_1.up3.up.weight.copy_(model_1.up3.up.weight)
    ref_1.up3.up.bias.copy_(model_1.up3.up.bias)
    copy_dc(model_1.up3.conv, ref_1.up3.conv)
    ref_1.up4.up.weight.copy_(model_1.up4.up.weight)
    ref_1.up4.up.bias.copy_(model_1.up4.up.bias)
    copy_dc(model_1.up4.conv, ref_1.up4.conv)
    ref_1.outc.conv.weight.copy_(model_1.outc.weight)
    ref_1.outc.conv.bias.copy_(model_1.outc.bias)

    out_ours = model_1(dummy_input)
    out_ref = ref_1(dummy_input)

max_diff = torch.max(torch.abs(out_ours - out_ref)).item()
assert max_diff <= 1e-5, f"Forward diff exceeded: {max_diff}"
assert out_ours.shape == (2, 1, img_size, img_size), f"Output shape mismatch: {out_ours.shape}"

dummy_ckpt = {"model_state_dict": model_1.state_dict()}
ckpt_path = "temp_ckpt_unet.pth"
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

print("UNet Verification Summary:")
print(f"  Params Ref 1 (milesial):    {p_ref:,}")
print(f"  Params Ours 1 (Binary):     {p_1:,}")
print(f"  Params Ours 4 (Multi):      {p_4:,}")
print(f"  Max Logit Diff vs Ref:      {max_diff:.8e}")
print(f"  thop MACs:                  {thop_macs / 1e9:.3f} G")
print(f"  ptflops MACs:               {pt_macs / 1e9:.3f} G")
print(f"  fvcore FLOPs:               {fv_flops / 1e9:.3f} G")
print("  Strict Load:                SUCCESS")
print("  Status:                     PASS")

