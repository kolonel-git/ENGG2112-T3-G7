"""Target C (fleet), step 2: labels, models and evaluation for alarm-group forecasts (D-071).

    python scripts/target_c_fleet_run.py --phase fit     # train years -> validation tables, frozen thresholds
    python scripts/target_c_fleet_run.py --phase test    # single scoring of the test years with frozen models

Anchors are full hours t (SCADA interval end); features use data to t only. y = 1 if an alarm start
of the group falls in (t, t + H]. Anchors with a group alarm active at t, a horizon that leaves the
split block, or no features carry no label. Thresholds are chosen on validation at a false-alarm
budget (false alert-hours per turbine-month) and frozen; recall is over episodes; PR-AUC intervals
bootstrap turbines.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from sklearn.linear_model import LogisticRegression

HOUR = np.timedelta64(1, "h")


# ---------------------------------------------------------------- configuration and blocks
def load_cfg(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def split_blocks(cfg: dict[str, Any]) -> list[tuple[str, np.datetime64, np.datetime64]]:
    """Contiguous year runs per split as (split, start, end); the test block ends at data_end."""
    blocks = []
    for split in ("train", "val", "test"):
        years = sorted(cfg["splits"][f"{split}_years"])
        run = [years[0]]
        for y in years[1:] + [None]:
            if y is not None and y == run[-1] + 1:
                run.append(y)
                continue
            start = np.datetime64(f"{run[0]}-01-01T00:00")
            end = np.datetime64(f"{run[-1] + 1}-01-01T00:00")
            if split == "test":
                end = min(end, np.datetime64(pd.Timestamp(cfg["data_end"]).to_datetime64(), "m"))
            blocks.append((split, start, end))
            if y is not None:
                run = [y]
    return blocks


def configs(cfg: dict[str, Any]) -> list[tuple[str, int]]:
    tc = cfg["target_c_fleet"]
    return [(g, h) for g in tc["groups"] for h in tc["horizons_hours"]]


# ---------------------------------------------------------------- labels and episodes
def label_anchors(
    t: np.ndarray, starts: np.ndarray, act_on: np.ndarray, act_off: np.ndarray,
    horizon_h: int, block_start: np.ndarray, block_end: np.ndarray,
) -> np.ndarray:
    """int8 labels: 1/0, or -1 when unlabelled (active alarm, horizon leaves the block)."""
    h = horizon_h * HOUR
    y = np.zeros(len(t), dtype=np.int8)
    if len(starts):
        nxt = np.searchsorted(starts, t, side="right")
        has = nxt < len(starts)
        y[has] = (starts[nxt[has]] <= t[has] + h).astype(np.int8)
    if len(act_on):
        cummax_off = np.maximum.accumulate(act_off.astype("int64")).astype(act_off.dtype)
        j = np.searchsorted(act_on, t, side="right") - 1
        active = (j >= 0) & (cummax_off[np.maximum(j, 0)] > t)
        y[active] = -1
    outside = (t < block_start) | (t + h > block_end)
    y[outside] = -1
    return y


def episode_starts(starts: np.ndarray, merge_h: int) -> np.ndarray:
    """First start of each run of starts closer together than merge_h hours."""
    if len(starts) == 0:
        return starts
    gap = np.diff(starts) >= merge_h * HOUR
    return starts[np.concatenate([[True], gap])]


# ---------------------------------------------------------------- metrics
def ap_presorted(y: np.ndarray, w: np.ndarray, group_end: np.ndarray) -> float:
    """Average precision (sklearn's step definition) on rows sorted by descending score."""
    tp = np.cumsum(w * y)[group_end]
    fp = np.cumsum(w * (1 - y))[group_end]
    total = tp[-1]
    if total <= 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / total
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


class APBootstrap:
    """PR-AUC point value and turbine-bootstrap interval without re-sorting per resample."""

    def __init__(self, score: np.ndarray, y: np.ndarray, turbine_idx: np.ndarray) -> None:
        order = np.argsort(-score, kind="stable")
        s = score[order]
        self.y = y[order].astype(np.float64)
        self.turb = turbine_idx[order]
        self.group_end = np.flatnonzero(np.concatenate([s[1:] != s[:-1], [True]]))

    def value(self, mult: np.ndarray | None = None) -> float:
        w = np.ones_like(self.y) if mult is None else mult[self.turb].astype(np.float64)
        return ap_presorted(self.y, w, self.group_end)


def threshold_at_budget(score: np.ndarray, y: np.ndarray, n_labelled: int, budget: float,
                        hours_per_month: float) -> float:
    """Lowest threshold whose false alerts (score > thr on y = 0) stay within the budget."""
    allowed = int(np.floor(budget * n_labelled / hours_per_month))
    neg = np.sort(score[y == 0])[::-1]
    if allowed >= len(neg):
        return -np.inf
    return float(neg[allowed])


# ---------------------------------------------------------------- data assembly
def load_alarms(path: Path) -> dict[int, pd.DataFrame]:
    a = pd.read_parquet(path)
    return {int(k): v.sort_values("TimeOn").reset_index(drop=True) for k, v in a.groupby("StationNr")}


def hours_since_last(t: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Hours from the last start at or before t (inf when there is none); starts sorted."""
    out = np.full(len(t), np.inf)
    if len(starts):
        last = np.searchsorted(starts, t, side="right") - 1
        ok = last >= 0
        out[ok] = (t[ok] - starts[last[ok]]) / HOUR
    return out


def history_features(t: np.ndarray, al: pd.DataFrame, codes: list[int], cap: float) -> np.ndarray:
    out = np.empty((len(t), 2 * len(codes)), dtype=np.float32)
    for k, code in enumerate(codes):
        st = al.loc[al["Alarmcode"] == code, "TimeOn"].to_numpy("datetime64[ns]")
        st = np.sort(st)
        out[:, 2 * k] = np.minimum(hours_since_last(t, st), cap)
        hi = np.searchsorted(st, t, side="right")
        lo = np.searchsorted(st, t - 24 * HOUR, side="right")
        out[:, 2 * k + 1] = hi - lo
    return out


def assemble(cfg: dict[str, Any], root: Path, splits: tuple[str, ...]) -> dict[str, Any]:
    """Concatenate all turbines' anchors in the requested splits with labels and history."""
    tc = cfg["target_c_fleet"]
    blocks = [b for b in split_blocks(cfg) if b[0] in splits]
    alarms = load_alarms(root / "alarms.parquet")
    codes = [int(c) for c in cfg["alarm_feature_codes"]]
    files = sorted((root / "features").glob("*.parquet"))
    sensor_cols: list[str] = []
    for f in files:
        sensor_cols = sorted(set(sensor_cols) | {c for c in pq.read_schema(f).names if c.startswith("wtc_")})
    parts: dict[str, list[Any]] = {k: [] for k in ("t", "turb", "split", "X", "H")}
    labels: dict[tuple[str, int], list[np.ndarray]] = {k: [] for k in configs(cfg)}
    hist_group: dict[str, list[np.ndarray]] = {g: [] for g in tc["groups"]}
    episodes: dict[tuple[str, int], list[tuple[int, np.datetime64, str]]] = {k: [] for k in configs(cfg)}
    stations: list[int] = []
    for ti, f in enumerate(files):
        station = int(f.stem)
        stations.append(station)
        df = pd.read_parquet(f)
        t_all = df["anchor"].to_numpy("datetime64[ns]")
        keep = np.zeros(len(t_all), dtype=bool)
        b_start = np.full(len(t_all), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
        b_end = b_start.copy()
        split = np.empty(len(t_all), dtype=object)
        for name, s, e in blocks:
            m = (t_all >= s) & (t_all < e)
            keep |= m
            b_start[m], b_end[m], split[m] = s, e, name
        df = df.loc[keep]
        t = t_all[keep]
        b_start, b_end, split = b_start[keep], b_end[keep], split[keep]
        parts["t"].append(t)
        parts["turb"].append(np.full(len(t), ti, dtype=np.int16))
        parts["split"].append(split)
        parts["X"].append(df.reindex(columns=sensor_cols).to_numpy(np.float32))
        al = alarms.get(station, pd.DataFrame(columns=["TimeOn", "TimeOff", "Alarmcode"]))
        parts["H"].append(history_features(t, al, codes, float(tc["history_cap_hours"])))
        for g, gcodes in tc["groups"].items():
            ga = al[al["Alarmcode"].isin(gcodes)]
            starts = np.sort(ga["TimeOn"].to_numpy("datetime64[ns]"))
            act = ga.dropna(subset=["TimeOff"]).sort_values("TimeOn")
            act_on = act["TimeOn"].to_numpy("datetime64[ns]")
            act_off = act["TimeOff"].to_numpy("datetime64[ns]")
            since = hours_since_last(t, starts)
            hist_group[g].append((1.0 / (1.0 + since)).astype(np.float32))
            for h in tc["horizons_hours"]:
                if (g, h) not in labels:
                    continue
                labels[(g, h)].append(label_anchors(t, starts, act_on, act_off, h, b_start, b_end))
                for e in episode_starts(starts, h):
                    for name, s, end in blocks:
                        if s < e <= end:
                            episodes[(g, h)].append((ti, e, name))
        print(f"{station}: {len(t):,} anchors", flush=True)
    return {
        "t": np.concatenate(parts["t"]), "turb": np.concatenate(parts["turb"]),
        "split": np.concatenate(parts["split"]), "X": np.vstack(parts["X"]),
        "H": np.vstack(parts["H"]), "sensor_cols": sensor_cols, "codes": codes,
        "labels": {k: np.concatenate(v) for k, v in labels.items()},
        "persist": {g: np.concatenate(v) for g, v in hist_group.items()},
        "episodes": episodes, "stations": stations,
    }


# ---------------------------------------------------------------- episode metrics
def episode_index(data: dict[str, Any], key: tuple[str, int], split: str) -> list[tuple[int, int, int, np.datetime64]]:
    """(turbine, lo, hi, start) row ranges [e - H, e) of each episode in the split.

    Rows of `data` are ordered by turbine, then time (one split per assembled table)."""
    h = key[1] * HOUR
    t, turb = data["t"], data["turb"]
    out = []
    bounds = np.searchsorted(turb, np.arange(len(data["stations"]) + 1))
    for ti, e, name in data["episodes"][key]:
        if name != split:
            continue
        a, b = bounds[ti], bounds[ti + 1]
        lo = a + np.searchsorted(t[a:b], e - h, side="left")
        hi = a + np.searchsorted(t[a:b], e, side="left")
        out.append((ti, lo, hi, e))
    return out


def episode_metrics(alert: np.ndarray, labelled: np.ndarray, t: np.ndarray,
                    eps: list[tuple[int, int, int, np.datetime64]], n_turb: int) -> dict[str, Any]:
    cs_l = np.concatenate([[0], np.cumsum(labelled)])
    al = alert & labelled
    cs_a = np.concatenate([[0], np.cumsum(al)])
    det = np.zeros(n_turb)
    ev = np.zeros(n_turb)
    leads = []
    for ti, lo, hi, e in eps:
        if cs_l[hi] - cs_l[lo] == 0:
            continue
        ev[ti] += 1
        if cs_a[hi] - cs_a[lo] > 0:
            det[ti] += 1
            first = lo + int(np.argmax(al[lo:hi]))
            leads.append((e - t[first]) / HOUR)
    return {"det": det, "ev": ev, "lead": float(np.median(leads)) if leads else float("nan")}


def evaluate(score: np.ndarray, y: np.ndarray, turb: np.ndarray, t: np.ndarray,
             eps: list, thr: dict[float, float], cfg: dict[str, Any], rng: np.random.Generator,
             n_turb: int) -> dict[str, Any]:
    tc = cfg["target_c_fleet"]
    lab = y >= 0
    s_l, y_l, tb_l = score[lab], y[lab], turb[lab]
    boot = APBootstrap(s_l, y_l, tb_l)
    n_res = int(cfg["evaluation"]["bootstrap_resamples"])
    alpha = (1 - float(cfg["evaluation"]["ci_level"])) / 2
    mults = np.stack([np.bincount(rng.integers(0, n_turb, n_turb), minlength=n_turb) for _ in range(n_res)])
    aps = np.array([boot.value(m) for m in mults])
    row: dict[str, Any] = {
        "n_labelled": int(lab.sum()), "prevalence": float(y_l.mean()),
        "pr_auc": boot.value(), "pr_auc_lo": float(np.nanquantile(aps, alpha)),
        "pr_auc_hi": float(np.nanquantile(aps, 1 - alpha)),
    }
    n_months = lab.sum() / float(tc["hours_per_month"])
    for budget, th in thr.items():
        alert = score > th
        em = episode_metrics(alert, lab, t, eps, n_turb)
        rec_b = [(m @ em["det"]) / max(m @ em["ev"], 1) for m in mults]
        fp = int((alert & lab & (y == 0)).sum())
        tag = f"b{budget:g}"
        row.update({
            f"{tag}_threshold": th, f"{tag}_episodes": int(em["ev"].sum()),
            f"{tag}_detected": int(em["det"].sum()),
            f"{tag}_recall": float(em["det"].sum() / max(em["ev"].sum(), 1)),
            f"{tag}_recall_lo": float(np.quantile(rec_b, alpha)),
            f"{tag}_recall_hi": float(np.quantile(rec_b, 1 - alpha)),
            f"{tag}_lead_h": em["lead"], f"{tag}_fa_rate": fp / n_months,
        })
    return row


# ---------------------------------------------------------------- preprocessing and models
def fit_preprocess(x: np.ndarray, names: list[str]) -> dict[str, Any]:
    """Train-only column filter (observed share >= 0.5, not constant), medians, mean and std."""
    idx, med, mean, std = [], [], [], []
    for j in range(x.shape[1]):
        col = x[:, j]
        fin = np.isfinite(col)
        if fin.mean() < 0.5:
            continue
        v = col[fin]
        if v.std() == 0:
            continue
        m = float(np.median(v))
        filled_mean = (v.sum() + m * (~fin).sum()) / len(col)
        filled_var = (((v - filled_mean) ** 2).sum() + (~fin).sum() * (m - filled_mean) ** 2) / len(col)
        idx.append(j)
        med.append(m)
        mean.append(filled_mean)
        std.append(np.sqrt(filled_var))
    return {"idx": np.array(idx), "names": [names[i] for i in idx],
            "median": np.array(med, dtype=np.float32), "mean": np.array(mean, dtype=np.float32),
            "std": np.array(std, dtype=np.float32)}


def transform(x: np.ndarray, pp: dict[str, Any], scale: bool) -> np.ndarray:
    """Select the kept columns and fill missing values with train medians (one copy)."""
    xs = x[:, pp["idx"]]
    for k in range(xs.shape[1]):
        col = xs[:, k]
        col[~np.isfinite(col)] = pp["median"][k]
        if scale:
            col -= pp["mean"][k]
            col /= pp["std"][k] if pp["std"][k] > 0 else 1.0
    return xs


def balanced_weights(y: np.ndarray) -> np.ndarray:
    n, n1 = len(y), max(int(y.sum()), 1)
    n0 = max(n - n1, 1)
    return np.where(y == 1, n / (2 * n1), n / (2 * n0)).astype(np.float32)


# ---------------------------------------------------------------- phases
def scale_rows(x: np.ndarray, pp: dict[str, Any]) -> np.ndarray:
    return ((x - pp["mean"]) / np.where(pp["std"] > 0, pp["std"], 1.0)).astype(np.float32)


# ---------------------------------------------------------------- resumable stages
# The machine holding the raw data stops any process after a few minutes, so every unit of work
# writes its result atomically and is skipped when the result exists. Re-run until "all done".
MODELS = ("constant", "random", "P1_persistence", "logistic", "lgbm_sensors", "lgbm_sensors_hist")
LEARNED = ("logistic", "lgbm_sensors", "lgbm_sensors_hist")


def save_atomic(path: Path, write: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)


def np_save(path: Path, arr: np.ndarray) -> None:
    def w(tmp: Path) -> None:
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
    save_atomic(path, w)


def stage_prep(cfg: dict[str, Any], root: Path, cache: Path, split: str) -> None:
    """Assemble one split, fit (train) or apply the preprocessing, cache arrays."""
    d = cache / split
    if (d / "done").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    data = assemble(cfg, root, (split,))
    names = data["sensor_cols"]
    hnames = [f"hist_{c}_{k}" for c in data["codes"] for k in ("hours_since", "n24h")]
    if split == "train":
        pre = {"pp_s": fit_preprocess(data["X"], names), "pp_hh": fit_preprocess(data["H"], hnames),
               "sensor_cols": names, "codes": data["codes"], "stations": data["stations"]}
        joblib.dump(pre, cache / "preprocess.joblib")
    else:
        pre = joblib.load(cache / "preprocess.joblib")
        if names != pre["sensor_cols"] or data["stations"] != pre["stations"]:
            raise SystemExit(f"{split}: sensor columns or turbines differ from train")
    xs = transform(data["X"], pre["pp_s"], scale=False)
    del data["X"]
    np_save(d / "xs.npy", xs)
    np_save(d / "xh.npy", np.hstack([xs, transform(data["H"], pre["pp_hh"], scale=False)]))
    np_save(d / "t.npy", data["t"])
    np_save(d / "turb.npy", data["turb"])
    for key, y in data["labels"].items():
        np_save(d / f"y_{key[0]}_{key[1]}.npy", y)
    for g, v in data["persist"].items():
        np_save(d / f"persist_{g}.npy", v)
    joblib.dump(data["episodes"], d / "episodes.joblib")
    (d / "done").write_text("ok")
    print(f"prep {split}: {len(data['t']):,} anchors, {xs.shape[1]} sensor columns kept", flush=True)


def fit_unit(cfg: dict[str, Any], cache: Path, out: Path, g: str, h: int, model: str) -> None:
    tc = cfg["target_c_fleet"]
    seed = int(cfg["seed"])
    target = out / "models" / f"{g}_{h}h_{model}.joblib"
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    y = np.load(cache / "train" / f"y_{g}_{h}.npy")
    lab = np.flatnonzero(y >= 0)
    if y[lab].sum() == 0:
        joblib.dump(None, target)
        return
    if model == "logistic":
        pre = joblib.load(cache / "preprocess.joblib")
        n = min(len(lab), int(tc["logistic_max_rows"]))
        pick = np.sort(np.random.default_rng(seed + 17 * h + len(g)).choice(lab, n, replace=False))
        x = np.load(cache / "train" / "xs.npy", mmap_mode="r")
        fitted: Any = LogisticRegression(C=float(tc["logistic_c"]), class_weight="balanced", max_iter=1000)
        fitted.fit(scale_rows(np.asarray(x[pick]), pre["pp_s"]), y[pick])
    else:
        name = "xs" if model == "lgbm_sensors" else "xh"
        x = np.load(cache / "train" / f"{name}.npy", mmap_mode="r")
        sub = lgb.Dataset(np.asarray(x[lab]), label=y[lab].astype(np.float32),
                          weight=balanced_weights(y[lab]), free_raw_data=True,
                          params={"verbose": -1, "max_bin": 63}).construct()
        params = {"objective": "binary", "verbose": -1, "num_threads": 2, "seed": seed, "max_bin": 63,
                  **{k: v for k, v in tc["lgbm"].items() if k != "n_estimators"}}
        booster = lgb.train(params, sub, num_boost_round=int(tc["lgbm"]["n_estimators"]))
        fitted = booster.model_to_string()
    joblib.dump(fitted, target)
    print(f"fit {g} {h}h {model}: {time.perf_counter() - began:.0f}s", flush=True)


def score_unit(cfg: dict[str, Any], cache: Path, out: Path, split: str, g: str, h: int, model: str) -> None:
    seed = int(cfg["seed"])
    target = out / "scores" / split / f"{g}_{h}h_{model}.npy"
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    d = cache / split
    n = len(np.load(d / "turb.npy"))
    if model == "constant":
        sc = np.zeros(n)
    elif model == "random":
        offset = 0 if split == "val" else 7
        sc = np.random.default_rng(seed + 1000 * h + len(g) + offset).random(n)
    elif model == "P1_persistence":
        sc = np.load(d / f"persist_{g}.npy")
    else:
        fitted = joblib.load(out / "models" / f"{g}_{h}h_{model}.joblib")
        if fitted is None:
            return
        if model == "logistic":
            pre = joblib.load(cache / "preprocess.joblib")
            sc = fitted.decision_function(scale_rows(np.load(d / "xs.npy"), pre["pp_s"]))
        else:
            x = np.load(d / ("xs.npy" if model == "lgbm_sensors" else "xh.npy"))
            sc = lgb.Booster(model_str=fitted).predict(x, num_threads=2)
    np_save(target, np.asarray(sc, dtype=np.float64))


def eval_unit(cfg: dict[str, Any], cache: Path, out: Path, split: str, g: str, h: int, model: str) -> None:
    tc = cfg["target_c_fleet"]
    seed = int(cfg["seed"])
    target = out / "rows" / split / f"{g}_{h}h_{model}.json"
    sc_path = out / "scores" / split / f"{g}_{h}h_{model}.npy"
    if target.exists() or not sc_path.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    d = cache / split
    sc = np.load(sc_path)
    y = np.load(d / f"y_{g}_{h}.npy")
    data = {"t": np.load(d / "t.npy"), "turb": np.load(d / "turb.npy"),
            "episodes": joblib.load(d / "episodes.joblib"),
            "stations": joblib.load(cache / "preprocess.joblib")["stations"]}
    lab = y >= 0
    frozen_path = out / "frozen_thresholds.json"
    key = f"{g}|{h}|{model}"
    if split == "val":
        thr = {float(b): threshold_at_budget(sc[lab], y[lab], int(lab.sum()), float(b),
                                             float(tc["hours_per_month"]))
               for b in tc["budgets_per_turbine_month"]}
        if model == "constant":
            thr = {b: 0.0 for b in thr}
    else:
        thr = {float(b): v for b, v in json.loads(frozen_path.read_text())[key].items()}
    row = evaluate(sc, y, data["turb"], data["t"], episode_index(data, (g, h), split), thr, cfg,
                   np.random.default_rng(seed), len(data["stations"]))
    row.update({"group": g, "horizon_h": h, "model": model, "split": split, "thresholds": thr})
    if split == "val":
        ytr = np.load(cache / "train" / f"y_{g}_{h}.npy")
        row.update({"train_labelled": int((ytr >= 0).sum()), "train_positive": int((ytr == 1).sum())})
    save_atomic(target, lambda tmp: tmp.write_text(json.dumps(row, default=float)))


def collect(cfg: dict[str, Any], out: Path, split: str) -> bool:
    files = sorted((out / "rows" / split).glob("*.json"))
    rows = [json.loads(f.read_text()) for f in files]
    expected = [(g, h, m) for g, h in configs(cfg) for m in MODELS]
    have = {(r["group"], r["horizon_h"], r["model"]) for r in rows}
    missing = [k for k in expected if k not in have]
    if split == "val":
        frozen = {f"{r['group']}|{r['horizon_h']}|{r['model']}": r["thresholds"] for r in rows}
        save_atomic(out / "frozen_thresholds.json", lambda tmp: tmp.write_text(json.dumps(frozen, indent=1)))
    table = pd.DataFrame(rows).drop(columns="thresholds")
    name = "validation.csv" if split == "val" else "test.csv"
    save_atomic(out / name, lambda tmp: table.to_csv(tmp, index=False))
    return not missing


def run(cfg: dict[str, Any], root: Path, out: Path, cache: Path, phase: str, budget_s: float) -> None:
    began = time.perf_counter()
    cache.mkdir(parents=True, exist_ok=True)

    def time_left(need: float) -> bool:
        return time.perf_counter() - began + need < budget_s

    if phase == "fit":
        for split in ("train", "val"):
            stage_prep(cfg, root, cache, split)
        for g, h in configs(cfg):
            for m in LEARNED:
                if not (out / "models" / f"{g}_{h}h_{m}.joblib").exists():
                    if not time_left(45 if m != "logistic" else 20):
                        print("pause: time budget", flush=True)
                        return
                    fit_unit(cfg, cache, out, g, h, m)
        units = [(g, h, m) for g, h in configs(cfg) for m in MODELS]
        for g, h, m in units:
            if not time_left(20):
                print("pause: time budget", flush=True)
                return
            score_unit(cfg, cache, out, "val", g, h, m)
            eval_unit(cfg, cache, out, "val", g, h, m)
        print("all done" if collect(cfg, out, "val") else "val rows missing", flush=True)
    else:
        if (out / "test.csv").exists() and collect(cfg, out, "test"):
            print("all done (test already scored)", flush=True)
            return
        if not (out / "frozen_thresholds.json").exists():
            raise SystemExit("run --phase fit to completion first")
        stage_prep(cfg, root, cache, "test")
        for g, h in configs(cfg):
            for m in MODELS:
                if not time_left(25):
                    print("pause: time budget", flush=True)
                    return
                score_unit(cfg, cache, out, "test", g, h, m)
                eval_unit(cfg, cache, out, "test", g, h, m)
        print("all done" if collect(cfg, out, "test") else "test rows missing", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["fit", "test"], required=True)
    ap.add_argument("--root", type=Path, default=Path("data/interim/target_c_fleet"))
    ap.add_argument("--out", type=Path, default=Path("reports/target_c_fleet"))
    ap.add_argument("--config", type=Path, default=Path("configs/target_c_fleet.yaml"))
    ap.add_argument("--cache", type=Path, default=None, help="model cache (default <root>/model_cache)")
    ap.add_argument("--budget-seconds", type=float, default=1e9)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.cache if args.cache is not None else args.root / "model_cache"
    run(cfg, args.root, args.out, cache, args.phase, args.budget_seconds)


if __name__ == "__main__":
    main()
