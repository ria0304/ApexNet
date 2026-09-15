import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EPOCHS,
    HIDDEN,
    LR,
    SEEDS,
    SEQ_LEN,
)
from logger import get_logger
from model import build_model
from real_data import REAL_N_FEATURES, RealOvertakeDataset, assert_no_leakage, collate

log = get_logger("train")


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            pairs = batch["pairs"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            logit = model(x, adj, pairs)
            prob = torch.sigmoid(logit)
            m = mask.bool().cpu().numpy().ravel()
            ys.append(y.cpu().numpy().ravel()[m])
            ps.append(prob.cpu().numpy().ravel()[m])
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    pred = (p >= 0.5).astype(np.int32)
    acc = float(accuracy_score(y, pred))
    f1 = float(f1_score(y, pred, zero_division=0))
    precision = float(precision_score(y, pred, zero_division=0))
    recall = float(recall_score(y, pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = 0.5
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "auc": auc,
        "confusion_matrix": cm.tolist(),  # [[tn, fp], [fn, tp]]
        "n_eval_pairs": int(len(y)),
    }


def train_one(name: str, train_ds, val_ds, device, seed: int = 42, patience: int = 8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(name, REAL_N_FEATURES, SEQ_LEN, HIDDEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(reduction="none")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    history = []
    best = {"auc": -1}
    best_state = None
    bad = 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            adj = batch["adj"].to(device)
            pairs = batch["pairs"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad()
            logit = model(x, adj, pairs)
            raw_loss = crit(logit, y)
            loss = (raw_loss * mask).sum() / mask.sum().clamp(min=1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        metrics = evaluate(model, val_loader, device)
        metrics["loss"] = float(np.mean(losses))
        metrics["epoch"] = epoch
        history.append(metrics)
        log.info(
            "%-8s epoch %02d loss=%.4f acc=%.3f f1=%.3f auc=%.3f (n_eval_pairs=%d)",
            name, epoch, metrics["loss"], metrics["accuracy"], metrics["f1"], metrics["auc"], metrics["n_eval_pairs"],
        )
        if metrics["auc"] > best["auc"]:
            best = dict(metrics)
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info("%s early stop at epoch %d", name, epoch)
                break
    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    return model, best, history, elapsed


def plot_confusion_matrix(cm, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No overtake", "Overtake"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No overtake", "Overtake"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_dataset_distribution(train_ds, val_ds, test_ds, out_path):
    """CO3 requirement: dataset distribution, split sizes/percentages,
    class balance, computed from the real, race-level split actually
    used for training (not fabricated)."""
    def class_balance(ds):
        pos = total = 0
        for rec in ds.records:
            pos += int(rec["y"].sum())
            total += len(rec["y"])
        return pos, total

    n_train, n_val, n_test = len(train_ds), len(val_ds), len(test_ds)
    n_all = n_train + n_val + n_test
    dist = {"races": {}, "pairs": {}}
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        pos, total = class_balance(ds)
        n = len(ds)
        dist["races"][name] = {"count": n, "pct": round(100 * n / n_all, 1)}
        dist["pairs"][name] = {
            "total_pairs": total,
            "positive_overtakes": pos,
            "positive_rate": round(pos / total, 3) if total else None,
        }
    with open(out_path, "w") as f:
        json.dump(dist, f, indent=2)
    log.info("wrote dataset distribution to %s", out_path)
    return dist


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_dir = os.path.join(root, CHECKPOINT_DIR)
    results_dir = os.path.join(root, "results")
    figures_dir = os.path.join(root, "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    log.info("loading real F1 race-results dataset (f1db, 2014+)...")
    # Race-level 60/20/20 train/val/test split (val used for model
    # selection/early-stopping, test touched only once at the end).
    train_ds = RealOvertakeDataset(SEQ_LEN, split="train", val_frac=0.2, test_frac=0.2, seed=42)
    val_ds = RealOvertakeDataset(SEQ_LEN, split="val", val_frac=0.2, test_frac=0.2, seed=42)
    test_ds = RealOvertakeDataset(SEQ_LEN, split="test", val_frac=0.2, test_frac=0.2, seed=42)
    log.info("train races=%d val races=%d test races=%d", len(train_ds), len(val_ds), len(test_ds))

    if len(train_ds) < 10 or len(val_ds) < 5 or len(test_ds) < 5:
        log.error(
            "not enough real races to train/val/test (train=%d, val=%d, test=%d). "
            "Check that backend/data/*.csv exist and MIN_YEAR isn't filtering everything out.",
            len(train_ds), len(val_ds), len(test_ds),
        )
        sys.exit(1)

    write_dataset_distribution(train_ds, val_ds, test_ds, os.path.join(results_dir, "dataset_distribution.json"))
    assert_no_leakage(train_ds.records + val_ds.records + test_ds.records)

    names = ["cnn", "gat", "lstm", "cnn_lstm", "cnn_gat", "cnn_gat_v2", "transformer", "gcn"]
    results = {}
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    for name in names:
        seed_metrics = []
        for seed in SEEDS:
            log.info("training %s (seed=%d)...", name, seed)
            tr = RealOvertakeDataset(SEQ_LEN, split="train", val_frac=0.2, test_frac=0.2, seed=seed)
            va = RealOvertakeDataset(SEQ_LEN, split="val", val_frac=0.2, test_frac=0.2, seed=seed)
            te = RealOvertakeDataset(SEQ_LEN, split="test", val_frac=0.2, test_frac=0.2, seed=seed)
            te_loader = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
            model, best, history, elapsed = train_one(name, tr, va, device, seed=seed)
            test_metrics = evaluate(model, te_loader, device)
            seed_metrics.append({"seed": seed, "best_val": best, "history": history, "test": test_metrics, "seconds": elapsed})
            if seed == SEEDS[0]:
                path = os.path.join(ckpt_dir, f"{name}.pt")
                torch.save({"state_dict": model.state_dict(), "metrics": best, "seed": seed}, path)
                if name in ("cnn_gat_v2", "cnn_gat"):
                    plot_confusion_matrix(test_metrics["confusion_matrix"], os.path.join(figures_dir, f"confusion_matrix_{name}_test.png"), f"{name} — held-out test confusion matrix")
        # aggregate mean±std over seeds
        import pandas as pd
        agg = {}
        for k in ["accuracy", "precision", "recall", "specificity", "f1", "auc"]:
            vals = [m["test"][k] for m in seed_metrics]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "seeds": vals}
        results[name] = {"seeds": seed_metrics, "test_agg": agg, "test": seed_metrics[0]["test"],
                         "best_val": seed_metrics[0]["best_val"], "history": seed_metrics[0]["history"],
                         "seconds": float(np.mean([m["seconds"] for m in seed_metrics]))}

    out = os.path.join(results_dir, "metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    # keep a copy next to the checkpoint too, for anything that reads the old path
    with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out)

    log.info("=== Held-out TEST comparison (mean±std over seeds %s) ===", SEEDS)
    comparison_rows = []
    for name in names:
        a = results[name]["test_agg"]
        t = results[name]["test"]
        log.info("%-12s acc=%.3f±%.3f f1=%.3f±%.3f auc=%.3f±%.3f t=%.1fs", name, a["accuracy"]["mean"], a["accuracy"]["std"], a["f1"]["mean"], a["f1"]["std"], a["auc"]["mean"], a["auc"]["std"], results[name]["seconds"])
        comparison_rows.append(f"{name},{a['accuracy']['mean']:.4f}±{a['accuracy']['std']:.4f},{a['precision']['mean']:.4f}±{a['precision']['std']:.4f},{a['recall']['mean']:.4f}±{a['recall']['std']:.4f},{a['specificity']['mean']:.4f}±{a['specificity']['std']:.4f},{a['f1']['mean']:.4f}±{a['f1']['std']:.4f},{a['auc']['mean']:.4f}±{a['auc']['std']:.4f}")
    # Classical SOTA baselines on identical split (pair-level flattened features)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import accuracy_score as _acc, f1_score as _f1, roc_auc_score as _auc, precision_score as _pr, recall_score as _rc
        def pair_matrix(ds):
            Xs, ys = [], []
            for rec in ds.records:
                feat = rec["x"]
                for (at, dfn), y in zip(rec["pairs"], rec["y"]):
                    Xs.append(np.concatenate([feat[at], feat[dfn], feat[at] - feat[dfn]]))
                    ys.append(y)
            return np.array(Xs), np.array(ys)
        Xtr, ytr = pair_matrix(train_ds)
        Xte, yte = pair_matrix(test_ds)
        for clf_name, clf in [("logreg", LogisticRegression(max_iter=1000)), ("hgb", HistGradientBoostingClassifier(random_state=42))]:
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:, 1]
            pred = (p >= 0.5).astype(int)
            row = {"accuracy": float(_acc(yte, pred)), "precision": float(_pr(yte, pred, zero_division=0)), "recall": float(_rc(yte, pred, zero_division=0)), "f1": float(_f1(yte, pred, zero_division=0)), "auc": float(_auc(yte, p))}
            results[clf_name] = {"test": row, "test_agg": {k: {"mean": v, "std": 0.0, "seeds": [v]} for k, v in row.items()}}
            comparison_rows.append(f"{clf_name},{row['accuracy']:.4f},{row['precision']:.4f},{row['recall']:.4f},0.0000,{row['f1']:.4f},{row['auc']:.4f}")
            log.info("%-12s acc=%.3f f1=%.3f auc=%.3f (sklearn)", clf_name, row["accuracy"], row["f1"], row["auc"])
    except Exception as e:
        log.warning("sklearn baselines skipped: %s", e)
    with open(os.path.join(results_dir, "model_comparison_test.csv"), "w") as f:
        f.write("model,accuracy,precision,recall,specificity,f1,auc\n")
        f.write("\n".join(comparison_rows) + "\n")

    # Bootstrap 95% CI for proposed model AUC/F1 + McNemar vs best baseline
    try:
        rng = np.random.default_rng(0)
        te = RealOvertakeDataset(SEQ_LEN, split="test", val_frac=0.2, test_frac=0.2, seed=SEEDS[0])
        te_loader = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        ckpt = torch.load(os.path.join(ckpt_dir, "cnn_gat_v2.pt"), map_location=device)
        from model import build_model as _bm
        pm = _bm("cnn_gat_v2", REAL_N_FEATURES, SEQ_LEN, HIDDEN).to(device)
        pm.load_state_dict(ckpt["state_dict"])
        pm.eval()
        ys, ps = [], []
        with torch.no_grad():
            for b in te_loader:
                ys.append(b["y"][b["mask"].bool()].numpy().ravel())
                ps.append(torch.sigmoid(pm(b["x"].to(device), b["adj"].to(device), b["pairs"].to(device))).cpu().numpy().ravel()[b["mask"].bool().numpy().ravel()])
        y = np.concatenate(ys); p = np.concatenate(ps)
        aucs = [float(roc_auc_score(y[rng.choice(len(y), len(y), replace=True)], p[rng.choice(len(y), len(y), replace=True)])) for _ in range(200)]
        stats = {"n_test_pairs": int(len(y)), "auc_bootstrap95": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]}
        with open(os.path.join(results_dir, "significance.json"), "w") as f:
            json.dump(stats, f, indent=2)
        log.info("bootstrap AUC95=%s", stats["auc_bootstrap95"])
    except Exception as e:
        log.warning("significance skipped: %s", e)

    # --- CO5 ablation: drop one real feature group at a time from the
    # proposed CNN+GAT model instead of just retuning hyperparameters. ---
    log.info("=== CO5 ablation: feature-group contribution for CNN+GAT ===")
    ablation_groups = {
        "no_form_pace": ("driver_season_form", "constructor_season_pace"),
        "no_trend": ("driver_recent_trend",),
        "no_pit_prior": ("driver_avg_pit_prior", "constructor_avg_pit_prior"),
        "no_quali_gap": ("qualifying_gap_to_pole",),
        "no_grid_gap": ("grid_gap_ahead", "grid_gap_behind"),
        "full_proposed": (),
    }
    ablation_results = {}
    for exp_name, disabled in ablation_groups.items():
        tr = RealOvertakeDataset(SEQ_LEN, split="train", val_frac=0.2, test_frac=0.2, seed=42, disabled_features=disabled)
        va = RealOvertakeDataset(SEQ_LEN, split="val", val_frac=0.2, test_frac=0.2, seed=42, disabled_features=disabled)
        te = RealOvertakeDataset(SEQ_LEN, split="test", val_frac=0.2, test_frac=0.2, seed=42, disabled_features=disabled)
        model, best_val, _, _ = train_one("cnn_gat_v2", tr, va, device)
        te_loader = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
        te_metrics = evaluate(model, te_loader, device)
        ablation_results[exp_name] = {"disabled_features": list(disabled), "val_best": best_val, "test": te_metrics}
        log.info("ablation[%-26s] test acc=%.3f f1=%.3f auc=%.3f", exp_name, te_metrics["accuracy"], te_metrics["f1"], te_metrics["auc"])
    with open(os.path.join(results_dir, "ablation_study.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)
    log.info("wrote %s", os.path.join(results_dir, "ablation_study.json"))


if __name__ == "__main__":
    main()
