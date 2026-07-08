from ultralytics import YOLO
from training_utils import (
    PrepareDataset,
    GetModelYaml,
    GetLatestWeightsDir,
)
from training_utils import (
    dataset_yaml_path,
    coco_classes_file,
    training_task,
    experiment_name,
)
import os
import argparse
from export import Export


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("-l", "--load", type=str, default=None, help="Path to the model weights to load. Load the pretrained model")
    parser.add_argument("-r", "--learning-rate", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("-e", "--epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("-p", "--patience", type=int, default=50, help="Number of epochs triggering early stopping when no improvement")
    parser.add_argument("-s", "--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("-b", "--disable-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("-n", "--no-aug", action="store_true", help="Disable all data augmentations during training")

    args = parser.parse_args()

    print(f"Model weights initialized from: {args.load if args.load else 'scratch'}")
    print(f"Learning rate: {args.learning_rate}, Epochs: {args.epochs}, Patience: {args.patience}, Batch size: {args.batch_size}")

    PrepareDataset(coco_classes_file, dataset_yaml_path, training_task)

    if args.load is not None:
        if os.path.exists(args.load):
            model = YOLO(args.load)
        else:
            print(f"[ERROR] : Model {args.load} does not exists")
            exit(1)
    else:
        model = YOLO(GetModelYaml(training_task))  # Initialize model

    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if args.no_aug:
        # Match CondInst's augmentation: horizontal flip only. CondInst uses RandomFlip (horizontal,
        # p=0.5) plus a mild multi-scale resize; it does NOT use vertical flip, color jitter, rotation,
        # scale/shear jitter, or mosaic/mixup. In particular flipud=0.0 (vertical flip is harmful on
        # natural images like VOC). NOTE: CondInst's mild multi-scale (short side 640-800) has no clean
        # ultralytics equivalent -- the multi_scale knob (0.5-1.5x) is far more aggressive -- so we keep
        # single-scale imgsz=768 rather than introduce a mismatched, stronger augmentation.
        aug_params = dict(
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
            flipud=0.0, fliplr=0.5, bgr=0.0,
            mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
            erasing=0.0,
        )
    else:
        aug_params = dict(
            flipud=0.5,
            fliplr=0.5,
            scale=0.2,
            mosaic=0.0,  # Please set this to 0.0 TODO: Fix the issue with mosaic and keypoint detection
        )

    print(f"Augmentations: {'disabled' if args.no_aug else 'enabled'}")

    # CondInst trains SGD at lr0=0.01 for batch 16. Linear-scale the LR to the configured batch size
    # so the per-step update magnitude matches (batch 64 -> 0.04). -r sets the batch-16 base LR.
    lr0 = args.learning_rate * args.batch_size / 16
    print(f"Linear-scaled lr0: {lr0} (base {args.learning_rate} @ batch 16 -> batch {args.batch_size})")

    model.train(
        task=training_task,
        data="verdant.yaml",
        optimizer='SGD',
        lr0=lr0,
        lrf=0.01,             # CondInst held LR ~constant (step milestones > max_iter); a mild cosine
                              # decay is kept deliberately since this run is far longer (500 epochs).
        momentum=0.9,         # CondInst SOLVER.MOMENTUM (ultralytics default is 0.937)
        weight_decay=0.0001,  # CondInst SOLVER.WEIGHT_DECAY (ultralytics default is 0.0005)
        amp=False,            # FrozenBN backbone (freeze_bn) doesn't re-normalize activations, so fp16
                              # AMP can overflow -> NaN. Train in fp32 (detectron2 runs FrozenBN in fp32).
        epochs=args.epochs,
        imgsz=768,
        seed=1,
        batch=args.batch_size,
        name=experiment_name,
        device=[0, 1, 2, 3, 4, 5, 6, 7],
        patience=args.patience,
        **aug_params,
    )

    print("Training completed. Exporting the model by converting checkpoints to ONNX format...")
    latest_weights_dir = GetLatestWeightsDir()
    Export(f"{latest_weights_dir}/best.pt")
    Export(f"{latest_weights_dir}/last.pt")
