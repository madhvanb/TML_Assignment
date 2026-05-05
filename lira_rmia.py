import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms
from tqdm import tqdm


NUM_SHADOWS = 32
EPOCHS = 25
BATCH_SIZE = 128
EVAL_BATCH_SIZE = 512
LR = 0.05
NUM_WORKERS = 4
SEED = 0

NUM_CLASSES = 9
MEAN = [0.7406, 0.5331, 0.7059]
STD  = [0.1491, 0.1864, 0.1301]

ROOT = Path("/home/atml_team011/tml26-mia")
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SHADOW_DIR = OUTPUT_DIR / "shadow_lira_rmia" / f"aug_n{NUM_SHADOWS}_e{EPOCHS}_seed{SEED}"
OUTPUT_PATH = OUTPUT_DIR / "submission_lira_rmia_aug.csv"


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = self.imgs[i]
        if self.transform is not None:
            img = self.transform(img)
        return self.ids[i], img, self.labels[i]


class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, i):
        id_, img, label = super().__getitem__(i)
        return id_, img, label, self.membership[i]


class AugmentedDataset(Dataset):
    def __init__(self, base):
        self.base = base
        self.augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        return (item[0], self.augment(item[1]), *item[2:])


def load_data():
    pub  = torch.load(DATA_DIR / "pub.pt",  weights_only=False)
    priv = torch.load(DATA_DIR / "priv.pt", weights_only=False)
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    pub.transform  = transform
    priv.transform = transform
    return pub, priv


def merge_datasets(pub, priv):
    combined = MembershipDataset(transform=pub.transform)
    combined.ids        = list(pub.ids)        + list(priv.ids)
    combined.imgs       = list(pub.imgs)       + list(priv.imgs)
    combined.labels     = list(pub.labels)     + list(priv.labels)
    combined.membership = list(pub.membership) + [-1] * len(priv)
    return combined


def build_model():
    model = resnet18(weights=None)
    model.conv1  = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model.fc     = torch.nn.Linear(512, NUM_CLASSES)
    return model


def train_collate(batch):
    images = torch.stack([item[1] for item in batch])
    labels = torch.tensor([int(item[2]) for item in batch])
    return images, labels


def eval_collate(batch):
    ids      = torch.tensor([int(item[0]) for item in batch])
    images   = torch.stack([item[1] for item in batch])
    labels   = torch.tensor([int(item[2]) for item in batch])
    membership = torch.tensor([int(item[3]) for item in batch])
    return ids, images, labels, membership


def random_half(n, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_idx = np.sort(perm[: n // 2])
    in_mask = np.zeros(n, dtype=np.int8)
    in_mask[train_idx] = 1
    return train_idx, in_mask


def train_shadow(pool, train_idx, device):
    model = build_model().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    loader = DataLoader(
        Subset(AugmentedDataset(pool), train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collate,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(loader):.3f}")

    model.eval()
    return model


def get_signal(logits, labels):
    probs  = torch.softmax(logits, dim=1)
    p_true = probs.gather(1, labels.view(-1, 1)).squeeze(1).clamp(1e-6, 1 - 1e-6)
    return torch.log(p_true / (1.0 - p_true))


@torch.no_grad()
def collect_signals(model, dataset, device, desc=""):
    loader = DataLoader(
        dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=eval_collate
    )
    ids_list, labels_list, signals_list, mem_list = [], [], [], []
    model.eval()
    for ids, images, labels, mem in tqdm(loader, desc=desc):
        images = images.to(device)
        labels = labels.to(device).long()
        signals = get_signal(model(images), labels)
        ids_list.append(ids.numpy())
        labels_list.append(labels.cpu().numpy())
        signals_list.append(signals.cpu().numpy())
        mem_list.append(mem.numpy())
    return {
        "id":         np.concatenate(ids_list),
        "label":      np.concatenate(labels_list),
        "signal":     np.concatenate(signals_list),
        "membership": np.concatenate(mem_list),
    }


def lira_scores(target_signals, in_obs, out_obs):
    mu_in    = np.nanmean(in_obs,  axis=1)
    mu_out   = np.nanmean(out_obs, axis=1)
    sigma_in  = np.fmax(np.nanstd(in_obs,  axis=1), 1e-3)
    sigma_out = np.fmax(np.nanstd(out_obs, axis=1), 1e-3)

    log_p_in  = -0.5 * ((target_signals - mu_in)  / sigma_in)  ** 2 - np.log(sigma_in)
    log_p_out = -0.5 * ((target_signals - mu_out) / sigma_out) ** 2 - np.log(sigma_out)

    scores = log_p_in - log_p_out
    return np.nan_to_num(scores, nan=0.0, posinf=30.0, neginf=-30.0)


def rmia_rank(scores, labels, ref_scores, ref_labels):
    out = np.zeros(len(scores))
    for c in range(NUM_CLASSES):
        mask     = labels     == c
        ref_mask = ref_labels == c
        if ref_mask.sum() == 0:
            out[mask] = 0.5
            continue
        sorted_refs = np.sort(ref_scores[ref_mask])
        out[mask] = np.searchsorted(sorted_refs, scores[mask], side="right") / len(sorted_refs)
    return out


def tpr_at_fpr(membership, scores, fpr_thresh=0.05):
    fpr, tpr, _ = roc_curve(membership.astype(int), scores.astype(float))
    valid = tpr[fpr <= fpr_thresh]
    return float(valid.max()) if len(valid) else 0.0


def save_submission(ids, scores, path):
    scores = np.asarray(scores, dtype=np.float64)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    scores = np.clip(scores, 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "score"])
        for sid, sc in zip(ids, scores):
            writer.writerow([str(sid), sc])
    print(f"saved to {path}  ({len(ids)} rows)")


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  shadows: {NUM_SHADOWS}  epochs: {EPOCHS}")
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)

    # dataset loaded
    pub, priv = load_data()
    n_pub, n_priv = len(pub), len(priv)
    pool = merge_datasets(pub, priv)
    n_total = len(pool)
    print(f"pool: {n_total}  ({n_pub} pub + {n_priv} priv)")

    in_signals  = [[] for _ in range(n_total)]
    out_signals = [[] for _ in range(n_total)]

    # train or load shadow models
    for sid in range(NUM_SHADOWS):
        cache = SHADOW_DIR / f"shadow_{sid}.npz"

        if cache.exists():
            data = np.load(cache)
            in_mask = data["in_mask"]
            signals = data["signal"]
            print(f"[{sid+1}/{NUM_SHADOWS}] cached")
        else:
            print(f"[{sid+1}/{NUM_SHADOWS}] training")
            train_idx, in_mask = random_half(n_total, SEED + sid)
            shadow = train_shadow(pool, train_idx, device)
            results = collect_signals(shadow, pool, device, desc=f"shadow {sid}")
            signals = results["signal"]
            np.savez(cache, in_mask=in_mask, signal=signals)
            del shadow

        for i in range(n_total):
            if in_mask[i]:
                in_signals[i].append(float(signals[i]))
            else:
                out_signals[i].append(float(signals[i]))

    # target model
    print("\nloading target")
    target = build_model()
    target.load_state_dict(torch.load(DATA_DIR / "model.pt", map_location="cpu"))
    target = target.to(device).eval()

    target_out = collect_signals(target, pool, device, desc="target")
    target_signals  = target_out["signal"]
    pool_labels     = target_out["label"]
    pool_membership = target_out["membership"]
    pool_ids        = target_out["id"]

    in_obs  = np.full((n_total, NUM_SHADOWS), np.nan)
    out_obs = np.full((n_total, NUM_SHADOWS), np.nan)
    for i in range(n_total):
        in_obs[i,  :len(in_signals[i])]  = in_signals[i]
        out_obs[i, :len(out_signals[i])] = out_signals[i]

    scores = lira_scores(target_signals, in_obs, out_obs)

    pub_scores     = scores[:n_pub]
    pub_labels     = pool_labels[:n_pub]
    pub_membership = pool_membership[:n_pub]
    priv_scores = scores[n_pub:]
    priv_labels = pool_labels[n_pub:]
    priv_ids    = pool_ids[n_pub:]

    print(f"\nLiRA only    pub TPR@5%FPR: {tpr_at_fpr(pub_membership, pub_scores):.4f}")

    # RMIA ranking against pub non-members
    non_member_mask = pub_membership == 0
    ref_scores = pub_scores[non_member_mask]
    ref_labels = pub_labels[non_member_mask]

    pub_final  = rmia_rank(pub_scores,  pub_labels,  ref_scores, ref_labels)
    priv_final = rmia_rank(priv_scores, priv_labels, ref_scores, ref_labels)
    print(f"LiRA + RMIA  pub TPR@5%FPR: {tpr_at_fpr(pub_membership, pub_final):.4f}")

    save_submission(priv_ids, priv_final, OUTPUT_PATH)


if __name__ == "__main__":
    main()
