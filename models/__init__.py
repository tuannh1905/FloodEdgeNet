import os
import torch
import importlib
import inspect


def get_model(model_name, num_classes=1, seed=None, input_size=256, pretrained=False):
    """
    Initialize model with DETERMINISTIC weights.

    Args:
        model_name: Name of model architecture (e.g. flood_edgenet, unet, deeplabv3, etc.)
        num_classes: Number of output classes
        seed:        Random seed (REQUIRED for reproducibility)
        input_size:  Input image size (phải chia hết cho 16)
        pretrained:  Load pretrained backbone weights if supported (e.g. ImageNet for heavy baselines)
    """
    if input_size % 16 != 0:
        raise ValueError(
            f"input_size={input_size} phải chia hết cho 16 "
            f"(do các tầng Downsampling trong encoder)."
        )

    # Normalize model name aliases
    alias_map = {
        'floodedgenet': 'flood_edgenet',
        'fsenet': 'flood_edgenet',
    }
    canonical_name = alias_map.get(model_name.lower(), model_name)

    model_file = os.path.join(os.path.dirname(__file__), f'{canonical_name}.py')
    if not os.path.exists(model_file):
        available = [
            f.replace('.py', '')
            for f in os.listdir(os.path.dirname(__file__))
            if f.endswith('.py') and f != '__init__.py'
        ]
        raise ValueError(
            f"Model '{model_name}' not found.\n"
            f"Available: {', '.join(sorted(available))}"
        )

    # Deterministic weight init
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    module = importlib.import_module(f'models.{canonical_name}')

    sig = inspect.signature(module.build_model)
    kwargs = {'num_classes': num_classes}
    if 'input_size' in sig.parameters:
        kwargs['input_size'] = input_size
    if 'pretrained' in sig.parameters:
        kwargs['pretrained'] = pretrained

    model = module.build_model(**kwargs)

    if seed is not None:
        print(f"[OK] Model '{canonical_name}' initialized with seed {seed} (pretrained={pretrained})")

    return model
