import os
import argparse
import json
import torch
import numpy as np

from models import get_model
from utils.dataloader import get_dataloaders
from utils.metrics import calculate_all_metrics


def evaluate_cross_dataset(model_name, weights_path, source_dataset, target_datasets,
                           size=256, batch_size=4, num_classes=1, seed=42, output_dir='outputs'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 75)
    print("CROSS-DATASET GENERALIZATION BENCHMARK (ZERO-SHOT DOMAIN TRANSFER)")
    print("=" * 75)
    print(f"Model Architecture  : {model_name}")
    print(f"Trained Checkpoint  : {weights_path}")
    print(f"Source Training Domain: {source_dataset.upper()}")
    print(f"Target Test Domains : {[d.upper() for d in target_datasets]}")
    print(f"Inference Device    : {device}")
    print("=" * 75 + "\n")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    # Initialize model and load weights
    model = get_model(model_name, num_classes=num_classes, seed=seed, input_size=size)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    domain_results = {}

    for target_ds in target_datasets:
        print(f"[*] Evaluating transferability: {source_dataset.upper()} -> {target_ds.upper()}...")
        try:
            _, _, test_loader = get_dataloaders(
                dataset=target_ds, batch_size=batch_size, size=size,
                seed=seed, num_classes=num_classes, dataset_type=target_ds
            )
        except Exception as e:
            print(f"    [WARN] Skipping '{target_ds}' due to load error: {e}")
            continue

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

                if num_classes == 1:
                    preds = torch.sigmoid(outputs)
                else:
                    preds = torch.softmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(masks.cpu().numpy())

        metrics = calculate_all_metrics(all_preds, all_labels, num_classes)
        domain_results[target_ds] = metrics

        print(f"    mIoU:         {metrics['miou']:.4f}")
        print(f"    Dice (F1):    {metrics['dice']:.4f}")
        print(f"    Pixel Acc:    {metrics['pixel_accuracy']:.4f}")
        print(f"    Precision:    {metrics['precision']:.4f}")
        print(f"    Recall:       {metrics['recall']:.4f}")
        print(f"    Boundary IoU: {metrics['boundary_iou']:.4f}")
        print(f"    FWIoU:        {metrics['fwiou']:.4f}\n")

    # Display Consolidated Table
    print("\n" + "=" * 85)
    print("CROSS-DATASET GENERALIZATION RESULTS FOR PAPER")
    print("=" * 85)
    header = f"{'Domain Transfer':<28} | {'mIoU':<8} | {'Dice':<8} | {'Precision':<10} | {'Recall':<8} | {'B-IoU':<8} | {'FWIoU':<8}"
    print(header)
    print("-" * len(header))
    for target_ds, m in domain_results.items():
        transfer_name = f"{source_dataset} -> {target_ds}"
        print(f"{transfer_name:<28} | {m['miou']:<8.4f} | {m['dice']:<8.4f} | {m['precision']:<10.4f} | {m['recall']:<8.4f} | {m['boundary_iou']:<8.4f} | {m['fwiou']:<8.4f}")
    print("=" * 85 + "\n")

    # Generate LaTeX Table code
    print("LaTeX Table Snippet:")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Cross-Dataset Zero-Shot Generalization Results}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\hline")
    print(r"Domain Transfer & mIoU & Dice & Precision & Recall & Boundary-IoU & FWIoU \\")
    print(r"\hline")
    for target_ds, m in domain_results.items():
        src_clean = source_dataset.replace('_', r'\_')
        tgt_clean = target_ds.replace('_', r'\_')
        print(f"{src_clean} $\\rightarrow$ {tgt_clean} & {m['miou']:.4f} & {m['dice']:.4f} & {m['precision']:.4f} & {m['recall']:.4f} & {m['boundary_iou']:.4f} & {m['fwiou']:.4f} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model_name}_cross_eval_{source_dataset}.json")
    with open(out_file, 'w') as f:
        json.dump({
            'model': model_name,
            'source_dataset': source_dataset,
            'weights': weights_path,
            'results': domain_results
        }, f, indent=2)
    print(f"Results saved to: {out_file}")
    return domain_results


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Generalization Evaluation Tool")
    parser.add_argument('--model', type=str, default='flood_edgenet', help="Model name (e.g. flood_edgenet, unet, fast-scnn)")
    parser.add_argument('--weights', type=str, required=True, help="Path to trained checkpoint (.pth)")
    parser.add_argument('--source_dataset', type=str, default='floodkaggle', help="Dataset model was trained on")
    parser.add_argument('--target_datasets', type=str, nargs='+', default=['floodkaggle', 'floodscene', 'floodvn'],
                        help="Datasets to evaluate transferability on")
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    evaluate_cross_dataset(
        model_name=args.model,
        weights_path=args.weights,
        source_dataset=args.source_dataset,
        target_datasets=args.target_datasets,
        size=args.size,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

