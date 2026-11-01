import subprocess
import sys
import os
import time

models = [
    "dabnet",
    "deeplabv3",
    "edanet",
    "enet",
    "espnetv2",
    "fast-scnn",
    "segformer",
    "segnet",
    "stdc1",
    "ulite",
    "unet"
]

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []
all_passed = True

print("=" * 80)
print("FLOODEDGENET BASELINE CROSS-AUDIT RUNNER")
print(f"Total Models to Verify: {len(models)}")
print(f"Working Directory:      {root_dir}")
print(f"Python Executable:      {sys.executable}")
print("=" * 80)

for idx, model_name in enumerate(models, 1):
    script_path = os.path.join(root_dir, "audit", model_name, "verify.py")
    if not os.path.exists(script_path):
        print(f"[{idx}/{len(models)}] {model_name.upper()}: MISSING SCRIPT -> {script_path}")
        results.append({
            "model": model_name,
            "status": "MISSING",
            "code": -1,
            "duration": 0.0,
            "output": ""
        })
        all_passed = False
        continue

    print(f"\n--- [{idx}/{len(models)}] Running Verification: {model_name} ---")
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, script_path],
        cwd=root_dir,
        capture_output=True,
        text=True
    )
    t1 = time.time()
    duration = t1 - t0

    if proc.returncode == 0:
        print(f"STATUS: PASS (took {duration:.2f}s)")
        print(proc.stdout.strip())
        results.append({
            "model": model_name,
            "status": "PASS",
            "code": 0,
            "duration": duration,
            "output": proc.stdout.strip()
        })
    else:
        print(f"STATUS: FAIL (Exit Code: {proc.returncode}, took {duration:.2f}s)")
        if proc.stdout.strip():
            print("STDOUT:")
            print(proc.stdout.strip())
        if proc.stderr.strip():
            print("STDERR:")
            print(proc.stderr.strip())
        results.append({
            "model": model_name,
            "status": "FAIL",
            "code": proc.returncode,
            "duration": duration,
            "output": proc.stderr.strip()
        })
        all_passed = False

print("\n" + "=" * 80)
print("CROSS-AUDIT CONSOLIDATED SUMMARY")
print("=" * 80)
print(f"{'Model':<15} | {'Exit Code':<10} | {'Duration (s)':<12} | {'Audit Status':<12}")
print("-" * 55)

for res in results:
    print(f"{res['model']:<15} | {res['code']:<10} | {res['duration']:<12.2f} | {res['status']:<12}")

print("=" * 80)
if all_passed:
    print("ALL 11 BASELINE MODELS SUCCESSFULLY PASSED VERIFICATION!")
    sys.exit(0)
else:
    print("ERROR: ONE OR MORE MODELS FAILED AUDIT VERIFICATION!")
    sys.exit(1)

