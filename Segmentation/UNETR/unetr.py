import pandas as pd
import numpy as np
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.transforms import (LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged,
                              RandFlipd, RandRotate90d, EnsureTyped, ToTensord, Compose, AsDiscrete,
                              RandScaleIntensityd, RandShiftIntensityd, ClassesToIndicesd,
                              RandCropByLabelClassesd, DeleteItemsd, SpatialPadd)
from monai.data import (PersistentDataset, DataLoader, decollate_batch, pad_list_data_collate)
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from tqdm import tqdm

TARGET_SPACING = (1.0, 1.0, 1.0)
CLIP_RANGE_DICOM = (-150, 250)
IMAGE_SIZE = (512, 512, 512)
PATCH_SIZE = (96, 96, 96)
NUM_PATCHES = 4  #
NUM_WORKERS = min(8, os.cpu_count() - 2)  #
BATCH_SIZE_TRAINING = 6
BATCH_SIZE_VALIDATION = 1

NUM_CLASSES = None
RATIO = [1.0, 1.0, 2.0, 2.0, 2.0, 1.0]
LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
NUM_FOLDS = 5
# 0. Background
# 1. Liver
# 2. Spleen
# 3. Left Kidney
# 4. Right Kidney
# 5. Bowel
CLASS_LABELS_RATIC = ["Mean", "Liver", "Spleen", "Left Kidney", "Right Kidney", "Bowel"]

LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dynamo_config = getattr(torch, "_dynamo").config
dynamo_config.cache_size_limit = 64
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def build_dataset_ratic(path_root):
    dataset_list = []

    path_metadata_file = os.path.join(path_root, "train_series_meta.csv")
    path_segmentation_files = os.path.join(path_root, "segmentations")

    df = None
    entries = None

    try:
        if not os.path.exists(path_metadata_file):
            print(f"Error: Path {path_metadata_file} does not exist")
            return
        df = pd.read_csv(path_metadata_file)
    except PermissionError:
        print(f"Error: No access rights for '{path_metadata_file}'")
    except Exception as e:
        print(f"Error: {e}")

    try:
        if not os.path.exists(path_segmentation_files):
            print(f"Error: Path {path_segmentation_files} does not exist")
            return
        entries = os.listdir(path_segmentation_files)
    except PermissionError:
        print(f"Error: No access rights for '{path_segmentation_files}'")
    except Exception as e:
        print(f"Error: {e}")

    series_patient_id_dict = {}

    for _, row in df.iterrows():
        patient_id = int(row.get("patient_id", None))
        series_id = int(row.get("series_id", None))

        series_patient_id_dict[series_id] = patient_id

    for entry in entries:
        series_id = int(entry.removesuffix(".nii"))
        patient_id = series_patient_id_dict[series_id]

        path_mask = os.path.join(path_root, "segmentations", entry)
        path_image = os.path.join(path_root, "train_images", f"{patient_id}", f"{series_id}")

        dataset_list.append({"image": path_image, "label": path_mask})

    return dataset_list


def create_train_test_val_items(dataset, train_size=0.7, val_size=0.2, test_size=0.1):
    n = len(dataset)

    train_items = dataset[:int(n * train_size)]
    val_items = dataset[int(n * train_size):int(n * (train_size + val_size))]
    test_items = dataset[int(n * (train_size + val_size)):]
    return train_items, val_items, test_items


def get_transformer(mode, dataset, target_spacing=TARGET_SPACING, clip_range=CLIP_RANGE_DICOM, image_size=IMAGE_SIZE,
                    patch_size=PATCH_SIZE):
    load_ratic = [
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
    ]
    allways = [
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=clip_range[0], a_max=clip_range[1], b_min=0.0, b_max=1.0,
                             clip=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
    ]

    post = [
        EnsureTyped(keys=["image", "label"]),
        ToTensord(keys=["image", "label"])
    ]

    if mode == "train":
        train = [
            ClassesToIndicesd(keys=["label"], num_classes=NUM_CLASSES),
            RandCropByLabelClassesd(keys=["image", "label"], label_key="label", spatial_size=patch_size,
                                    num_classes=NUM_CLASSES,
                                    num_samples=NUM_PATCHES, ratios=RATIO),
            DeleteItemsd(keys=["label_cls_indices"]),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),  # Sagittal
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),  # Coronal
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),  # Axial
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            RandScaleIntensityd(keys=["image"], prob=0.2, factors=0.1),
            RandShiftIntensityd(keys=["image"], prob=0.2, offsets=0.1)
        ]
        return Compose(load_ratic + allways + train + post)  # allways + train + post
    else:
        return Compose(load_ratic + allways + post)


def train_one_epoch(model, optimizer, train_loader, loss_fn, epoch, scaler):
    global iteration
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} train")
    for iteration, batch in enumerate(progress_bar):
        imgs = batch["image"].to(DEVICE, non_blocking=True)
        labels = batch["label"].to(DEVICE, non_blocking=True)
        optimizer.zero_grad(
            set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item(), "iter": iteration})
    return epoch_loss / len(train_loader), (iteration + 1)


@torch.no_grad()
def evaluate(epoch, model, val_loader, dice_metric):
    model.eval()
    dice_metric.reset()
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} val")
    for batch in progress_bar:
        imgs = batch["image"].to(DEVICE)  # Shape: (B, 6, H, W, D)
        labels = batch["label"].to(DEVICE)  # Shape: (B, 1, H, W, D)
        predictions = sliding_window_inference(imgs, roi_size=PATCH_SIZE, sw_batch_size=BATCH_SIZE_VALIDATION,
                                               predictor=model,
                                               overlap=0.5, )  # Return Value Shape: (B, 6, H, W, D)

        postprocess_predictions = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
        postprocess_label = AsDiscrete(to_onehot=NUM_CLASSES)

        val_predictions = [postprocess_predictions(i) for i in decollate_batch(predictions)]
        val_labels = [postprocess_label(i) for i in decollate_batch(labels)]

        dice_metric(y_pred=val_predictions, y=val_labels)

    metric = dice_metric.aggregate(reduction="mean_batch")

    dice_metric.reset()
    return metric.cpu().numpy()


def set_up_model_and_train(train_items, val_items, num_epochs, dataset, save_path="", cache_path="TMPDIR"):
    hpc_tmp_dir = os.environ.get(cache_path, "/tmp")
    cache_dir = os.path.join(hpc_tmp_dir, "monai_cache")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Nutze lokalen Cache auf Node: {cache_dir}")
    train_transforms = get_transformer(dataset=dataset, mode="train")
    val_transforms = get_transformer(dataset=dataset, mode="val")

    train_dataset = PersistentDataset(data=train_items, transform=train_transforms, cache_dir=cache_dir)
    val_dataset = PersistentDataset(data=val_items, transform=val_transforms, cache_dir=cache_dir)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAINING, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=pad_list_data_collate, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE_VALIDATION, shuffle=False,
                            num_workers=NUM_WORKERS)

    model = UNETR(in_channels=1, out_channels=NUM_CLASSES, img_size=PATCH_SIZE, feature_size=16, hidden_size=768,
                  mlp_dim=3072, num_heads=12, proj_type="conv", norm_name="instance")

    model.to(DEVICE)

    scaler = torch.amp.GradScaler('cuda')
    model = torch.compile(model)

    try:
        state_dict = torch.load("best_unetr_ratic.pth", map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Load model: success")
    except:
        print("Load model: failed")

    all_dice_metrics = []
    try:
        load_array = np.load("all_dice_metrics.npy")
        all_dice_metrics = load_array.tolist()
        print("Load metrics: success")
        print(f"{all_dice_metrics}")
    except:
        print("Load metrics: failed")

    iterations = 0
    best_val = -1.0
    try:
        load_data = np.load("save_data.npy")
        data_array = load_data.tolist()
        iterations = data_array[0]
        best_val = data_array[1]
        print("Load Data: success")
    except:
        print("Load Data: failed")

    class_weights = torch.tensor(LOSS_WEIGHTS).to(DEVICE)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.1, patience=30, min_lr=1e-7)
    dice_metric = DiceMetric(include_background=False, reduction="none")

    for epoch in range(num_epochs):
        train_loss, epoch_iteration = train_one_epoch(model, optimizer, train_loader, loss_function, epoch, scaler)
        current_val_dice_vector = evaluate(epoch, model, val_loader, dice_metric)
        val_dice = np.mean(current_val_dice_vector)
        val_dice_vector = np.concatenate(([val_dice], current_val_dice_vector))
        all_dice_metrics.append(val_dice_vector)
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_dice={val_dice:.4f}, LR={current_lr:.2e}")
        if val_dice > best_val:
            best_val = val_dice
            torch.save(model.state_dict(), "best_unetr_ratic.pth")
            print(f"Saved best model with val_dice={best_val:.4f}")
        iterations += epoch_iteration

        print(f"Iterations: {iterations}")
        save_array = [iterations, best_val]
        try:
            save_data_np = np.array(save_array)
            np.save("save_data.npy", save_data_np)
            print("Save Data: success")
        except:
            print("Save Data: failed")

        try:
            all_dice_metrics_np = np.array(all_dice_metrics)
            np.save("all_dice_metrics.npy", all_dice_metrics_np)
            print("Save Metrics: success")
        except:
            print("Save Metrics: failed")

    return all_dice_metrics, model, dice_metric


def train(path_root, path_save, num_epochs, dataset):
    global NUM_CLASSES
    global RATIO
    global LOSS_WEIGHTS
    print(f"Device: {DEVICE}")
    NUM_CLASSES = 6
    RATIO = [1.0, 1.0, 2.0, 2.0, 2.0, 1.0]
    LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    dataset_dict = build_dataset_ratic(path_root)
    print(f"Size Training Dataset: {len(dataset_dict)}")
    train_items, val_items, test_items = create_train_test_val_items(dataset_dict)

    all_dice_metrics, model, _ = set_up_model_and_train(train_items, val_items, num_epochs, dataset, path_save)
