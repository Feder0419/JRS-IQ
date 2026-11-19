"""Step 4: cluster the front pages only, then health-check each cluster by OCR.

Only fronts are clustered (continuations just follow their front), each
front once, the right way up. Feature per page: PCA of the whole-page +
title-band fingerprint, plus an LDA projection learned from the pages
whose form code is already known (title OCR, overrides, human labels) -
LDA stretches exactly the directions that separate one form type from
another, which is what the raw fingerprint is too blunt to do. To keep
the purity check honest, each known page's LDA coordinates come from a
model fitted without it (5-fold cross-fitting).

k is picked as the smallest value whose clusters are >= TARGET_PURITY pure
on the known-code pages. Each cluster then gets a status:

Every member also gets a kNN-predicted code (nearest known-code pages in the
LDA-enhanced space), so clusters with no OCR hits still get a suggestion.

  ready         >= PRED_SHARE of members predicted the same code AND the
                known codes agree (>= READY_SHARE of >= MIN_CODED pages) ->
                suggested_code filled in, waiting for a human to confirm
  needs_review  members agree on a predicted code but too few known-code
                pages back it up - suggested_code is the kNN guess
  split         mixed; separated by predicted code into new clusters, accepted
                only if kNN gets >= SPLIT_ACCURACY of the known pages right
  pending       mixed and not cleanly separable -> every member goes to
                the pending pool

Members far from their cluster centre (beyond median + OUTLIER_MADS * MAD)
are also sent to the pending pool. Centres, thresholds and projections are
saved to store/front_model.pkl so later imports can be assigned directly.

Writes store/front_clusters.csv (one row per cluster, with form_code /
confirmed columns to fill in), store/front_pages.csv (one row per front),
and store/front_clusters/cluster_XXX/ sample images.

Usage:
    python cluster_fronts.py [--k auto] [--lda-weight 0.5]
"""
import argparse
import pickle
import shutil
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

import store
from forms import canon_code
from stage1_discover import DocCache, make_montage, save_full_images

HDR_WEIGHT = 2.0
N_COMPONENTS = 64
MIN_CLASS = 5
TARGET_PURITY = 0.99
READY_SHARE = 0.95
PRED_SHARE = 0.9
SUGGEST_SHARE = 0.5
MIN_CODED = 3
SPLIT_ACCURACY = 0.95
SPLIT_K = 7
NN_DIST_SLACK = 1.2
OUTLIER_MADS = 4.0
K_RANGE = range(4, 81, 2)
FULL_ZOOM = 2.0

CLUSTER_FIELDS = ["cluster_id", "n_members", "status", "suggested_code", "ocr_codes", "n_coded",
                  "purity", "split_from", "n_outliers", "files", "montage_file",
                  "form_code", "confirmed", "notes"]
PAGE_FIELDS = ["file", "page", "rotation", "cluster", "dist", "status", "known_code",
               "known_source", "suggested_code"]


def known_codes(fronts, labels, pages):
    out = []
    for f, p, rot in fronts:
        lab = labels.get((f, p))
        if lab and lab["source"] in ("override", "human") and lab["form_code"]:
            out.append((canon_code(lab["form_code"]), lab["source"]))
            continue
        pg = pages.get((f, p))
        code = pg and pg.get(f"title_code{rot}")
        out.append((code, "ocr") if code else (None, ""))
    return out


def lda_features(Z, codes):
    """-> (LDA coords for every row, fitted LDA, scale) with cross-fitted
    coordinates for the rows whose code is known."""
    counts = Counter(c for c in codes if c)
    classes = {c for c, n in counts.items() if n >= MIN_CLASS}
    known = np.array([c in classes for c in codes])
    if len(classes) < 2:
        return np.zeros((len(Z), 0)), None, None
    y = np.array([c if c in classes else "" for c in codes])
    lda = LinearDiscriminantAnalysis().fit(Z[known], y[known])
    L = lda.transform(Z)
    idx = np.where(known)[0]
    n_splits = min(5, min(counts[c] for c in classes))
    if n_splits >= 2:
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=0).split(idx, y[idx]):
            fold = LinearDiscriminantAnalysis().fit(Z[idx[tr]], y[idx[tr]])
            L[idx[te]] = fold.transform(Z[idx[te]])
    scale = L[~known].std(0) + 1e-9 if (~known).any() else L.std(0) + 1e-9
    return L / scale, lda, scale


def purity(labels_, codes):
    wrong = total = 0
    for c in set(labels_):
        cc = [codes[i] for i in np.where(labels_ == c)[0] if codes[i]]
        if cc:
            wrong += len(cc) - Counter(cc).most_common(1)[0][1]
            total += len(cc)
    return 1 - wrong / max(total, 1)


def choose_k(X, codes, k_arg):
    if k_arg != "auto":
        k = int(k_arg)
        return k, KMeans(k, n_init=10, random_state=0).fit(X)
    best = None
    print("   k  purity(known-code pages)")
    for k in K_RANGE:
        if k >= len(X):
            break
        km = KMeans(k, n_init=10, random_state=0).fit(X)
        pur = purity(km.labels_, codes)
        print(f"  {k:3d}  {pur:.4f}")
        if best is None or pur > best[1] + 1e-9:
            best = (k, pur, km)
        if pur >= TARGET_PURITY:
            return k, km
    print(f"  no k reached {TARGET_PURITY:.0%}; using the purest (k={best[0]})")
    return best[0], best[2]


def robust_threshold(d):
    med = np.median(d)
    mad = 1.4826 * np.median(np.abs(d - med)) + 1e-9
    return med + OUTLIER_MADS * mad


class CodeVoter:
    """kNN vote over the known-code pages. Refuses to answer (None) for a page
    further from its neighbours than known pages usually are from theirs -
    faded scans unlike any legible page would otherwise just inherit the
    least-far known code."""

    def __init__(self, X, codes):
        self.X, self.codes = X, codes
        self.ref = np.array([i for i, c in enumerate(codes) if c])
        self.ref_codes = np.array([codes[i] for i in self.ref])
        loo = [self._neighbours(i)[1] for i in self.ref]
        self.max_dist = float(np.percentile(loo, 95)) * NN_DIST_SLACK

    def _neighbours(self, m):
        d = np.linalg.norm(self.X[self.ref] - self.X[m], axis=1)
        d[self.ref == m] = np.inf
        nn = np.argsort(d)[:SPLIT_K]
        return nn, float(d[nn].mean())

    def vote(self, m):
        nn, dist = self._neighbours(m)
        if dist > self.max_dist:
            return None
        return str(Counter(self.ref_codes[nn]).most_common(1)[0][0])


def try_split(members, voter, codes):
    """-> ({member: code or None}, accuracy on the members' known codes or None)"""
    pred, hits, n = {}, 0, 0
    for m in members:
        vote = voter.vote(m)
        pred[m] = codes[m] or vote
        if codes[m]:
            n += 1
            hits += vote == codes[m]
    return pred, (hits / n if n else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", default="auto")
    ap.add_argument("--lda-weight", type=float, default=0.5)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite front_clusters.csv even if it has confirmed rows")
    args = ap.parse_args()

    keys, arrays = store.load_fingerprints()
    kidx = {k: i for i, k in enumerate(keys)}
    roles = store.load_roles()
    labels = store.load_labels()
    pages = store.load_pages()

    fronts = sorted((f, p, r["rotation"]) for (f, p), r in roles.items() if r["role"] == "front")
    rows = np.array([kidx[store.key(f, p)] for f, p, _ in fronts])
    rots = np.array([rot for _, _, rot in fronts])
    up = lambda kind: np.where((rots == 0)[:, None], arrays[f"{kind}0"][rows], arrays[f"{kind}180"][rows])
    raw = np.hstack([up("full"), HDR_WEIGHT * up("hdr")])
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(raw)
    Z = pca.transform(raw)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9

    kc = known_codes(fronts, labels, pages)
    codes = [c for c, _ in kc]
    print(f"{len(fronts)} front pages, {sum(1 for c in codes if c)} with a known code: "
          f"{dict(Counter(c for c in codes if c))}")
    L, lda, lda_scale = lda_features(Z, codes)
    X = np.hstack([Z, args.lda_weight * L])

    k, km = choose_k(X, codes, args.k)
    assign = km.labels_.copy()
    centers = {int(c): km.cluster_centers_[c] for c in range(k)}
    print(f"\nk = {k}, purity on known-code pages {purity(assign, codes):.4f}")

    voter = CodeVoter(X, codes)
    clusters, page_status, page_code = {}, {}, {}
    next_id = k
    for cid in range(k):
        members = list(np.where(assign == cid)[0])
        cc = Counter(codes[m] for m in members if codes[m])
        n_coded = sum(cc.values())
        pred, acc = try_split(members, voter, codes)
        groups = Counter(c for c in pred.values() if c)
        n_unknown = sum(1 for c in pred.values() if c is None)
        top_code, top_n = groups.most_common(1)[0] if groups else (None, 0)
        pred_share = top_n / len(members)
        coded_share = cc.most_common(1)[0][1] / n_coded if n_coded else 0.0
        info = dict(cluster_id=cid, ocr_codes=";".join(f"{c}:{n}" for c, n in cc.most_common()),
                    n_coded=n_coded, purity=round(coded_share, 3) if n_coded else "",
                    split_from="", suggested_code="",
                    notes=f"kNN: {dict(groups.most_common())}, too far to call: {n_unknown}")
        conflicting = len(groups) > 1 and groups.most_common(2)[1][1] >= max(2, 0.05 * len(members))
        if not conflicting and top_code and (n_coded == 0 or cc.most_common(1)[0][0] == top_code):
            agrees = (n_coded >= MIN_CODED and coded_share >= READY_SHARE
                      and pred_share >= PRED_SHARE)
            info.update(status="ready" if agrees else "needs_review",
                        suggested_code=top_code if pred_share >= SUGGEST_SHARE else "")
        elif not groups:
            info["status"] = "needs_review"
        else:
            if acc is not None and acc >= SPLIT_ACCURACY and len(groups) > 1:
                info.update(status="split", notes=f"split by code, kNN accuracy {acc:.0%}")
                for code, _ in groups.most_common():
                    sub = [m for m in members if pred[m] == code]
                    sc = Counter(codes[m] for m in sub if codes[m])
                    nc = sum(sc.values())
                    clusters[next_id] = dict(
                        cluster_id=next_id, members=sub, suggested_code=code,
                        status="ready" if sc[code] >= MIN_CODED else "needs_review",
                        ocr_codes=";".join(f"{c}:{n}" for c, n in sc.most_common()), n_coded=nc,
                        purity=round(sc[code] / nc, 3) if nc else "", split_from=cid)
                    centers[next_id] = X[sub].mean(0)
                    assign[sub] = next_id
                    next_id += 1
                # members too far from any known page stay behind, pending
                members = [m for m in members if pred[m] is None]
                for m in members:
                    page_status[m] = "pending"
            else:
                acc_txt = "no known pages to verify a split" if acc is None else f"split kNN accuracy only {acc:.0%}"
                info.update(status="pending", notes=f"mixed ({dict(groups.most_common())}), {acc_txt}")
                for m in members:
                    page_status[m] = "pending"
        info["members"] = members
        clusters[cid] = info

    thresholds = {}
    for cid, info in clusters.items():
        ms = info["members"]
        info["n_members"] = len(ms)
        if not ms:
            continue
        d = np.linalg.norm(X[ms] - centers[cid], axis=1)
        thresholds[cid] = float(robust_threshold(d)) if len(ms) >= 5 else float(d.max() * 1.5 + 1e-6)
        info["n_outliers"] = 0
        for m, dm in zip(ms, d):
            page_code[m] = info["suggested_code"]
            if info["status"] == "pending":
                continue
            if dm > thresholds[cid]:
                page_status[m] = "outlier"
                info["n_outliers"] += 1
            else:
                page_status.setdefault(m, "ok")

    # ---- outputs -------------------------------------------------------
    front_rows = []
    for m, (f, p, rot) in enumerate(fronts):
        cid = int(assign[m])
        front_rows.append(dict(file=f, page=p + 1, rotation=rot, cluster=cid,
                               dist=round(float(np.linalg.norm(X[m] - centers[cid])), 4),
                               status=page_status.get(m, "ok"), known_code=codes[m] or "",
                               known_source=kc[m][1], suggested_code=page_code.get(m, "")))
    store.write_csv(store.FRONT_PAGES_CSV, front_rows, PAGE_FIELDS)

    out_csv = store.FRONT_CSV
    existing = store.read_csv(out_csv)
    if any((r.get("confirmed") or "").strip() for r in existing) and not args.force:
        out_csv = store.FRONT_CSV.with_name("front_clusters.NEW.csv")
        print(f"WARNING: {store.FRONT_CSV.name} has confirmed rows - writing {out_csv.name} instead")

    cluster_rows = []
    for cid in sorted(clusters):
        info = clusters[cid]
        ms = info["members"]
        info["files"] = ";".join(sorted({fronts[m][0] for m in ms}))[:150]
        info["montage_file"] = f"front_clusters/cluster_{cid:03d}/thumbnail.png" if ms else ""
        info.setdefault("n_outliers", 0)
        info.setdefault("notes", "")
        info["form_code"] = info["confirmed"] = ""
        cluster_rows.append(info)
    store.write_csv(out_csv, cluster_rows, CLUSTER_FIELDS)

    with open(store.FRONT_MODEL, "wb") as fh:
        pickle.dump(dict(pca=pca, lda=lda, lda_scale=lda_scale, lda_weight=args.lda_weight,
                         hdr_weight=HDR_WEIGHT, centers=centers, thresholds=thresholds,
                         status={cid: c["status"] for cid, c in clusters.items()},
                         suggested={cid: c["suggested_code"] for cid, c in clusters.items()}), fh)

    if not args.no_images:
        if store.FRONT_DIR.exists():
            shutil.rmtree(store.FRONT_DIR)
        docs = DocCache(store.PDF_DIR)
        try:
            for cid, info in clusters.items():
                ms = info["members"]
                if not ms:
                    continue
                d = np.linalg.norm(X[ms] - centers[cid], axis=1)
                order = [ms[i] for i in np.argsort(d)]
                pick = order[:6] + order[6:][-3:]
                if info["status"] == "pending":  # show every code present
                    for code in {codes[m] for m in ms if codes[m]}:
                        pick += [m for m in order if codes[m] == code][:2]
                    pick = list(dict.fromkeys(pick))[:12]
                sample = [dict(file=fronts[m][0], page=fronts[m][1], rot=fronts[m][2],
                               dist=float(np.linalg.norm(X[m] - centers[cid]))) for m in pick]
                cdir = store.FRONT_DIR / f"cluster_{cid:03d}"
                cdir.mkdir(parents=True, exist_ok=True)
                make_montage(docs, sample, cdir / "thumbnail.png")
                save_full_images(docs, sample, cdir / "full", zoom=FULL_ZOOM)
        finally:
            docs.close_all()

    st = Counter(c["status"] for c in clusters.values() if c["members"])
    print(f"\nclusters with members: {sum(st.values())} -> {dict(st)}")
    print(f"front pages: {dict(Counter(r['status'] for r in front_rows))}")
    print(f"wrote {out_csv}, {store.FRONT_PAGES_CSV}, {store.FRONT_MODEL}")


if __name__ == "__main__":
    main()
