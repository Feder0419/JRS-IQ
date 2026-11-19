"""Step 3: decide every page's role (front / continuation / ignore) and rotation.

Reference set: every labelled page in store/page_labels.csv, fingerprinted the
right way up. Each page is compared, both ways up, against that set with a
k-nearest-neighbour vote (labelled pages leave themselves out), giving:

  - for unlabelled pages (new imports): the role and rotation to use; low
    vote confidence sends the page to the pending pool (status=pending)
  - for labelled pages: the label is kept, but a confident disagreement is
    flagged (status=role_conflict / rotation_conflict) - this is how wrong
    tags inside mixed clusters surface

Rotation: for fronts, whole-page OCR confidence (3439/3439 agree with the
labels); for other pages the kNN orientation (97.5% on labelled
continuations, 99.6% where decisive OCR agrees), and when decisive OCR
disagrees, the file's dominant continuation rotation + status=pending.

Writes store/roles.csv (every page) and store/role_review.csv (just the
pages needing a look).

Usage:
    python classify_roles.py [--k 15]
"""
import argparse
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import PCA

import store

ROLES = ["front", "continuation", "ignore"]
HDR_WEIGHT = 1.0
N_COMPONENTS = 128
CONF_OK = 0.8
FRONT_OCR_MARGIN = 0.2
OTHER_OCR_MARGIN = 0.4
CONFLICT_OCR_MARGIN = 0.6

FIELDS = ["file", "page", "role", "rotation", "status", "role_source", "orient_source",
          "label_role", "label_rotation", "knn_role", "knn_conf", "knn_rot",
          "ocr_rot", "ocr_margin", "title_code", "note"]


def page_vectors(arrays, rot):
    return np.hstack([arrays[f"full{rot}"], HDR_WEIGHT * arrays[f"hdr{rot}"]])


def knn_votes(Q, R, ref_roles, self_idx, k):
    """For each query row: ({role: share of weighted vote}, mean top-5 similarity)."""
    role_idx = np.array([ROLES.index(r) for r in ref_roles])
    probs = np.zeros((len(Q), len(ROLES)))
    top_sim = np.zeros(len(Q))
    for s in range(0, len(Q), 1000):
        sims = Q[s:s + 1000] @ R.T
        for j, ref in enumerate(self_idx[s:s + 1000]):
            if ref >= 0:
                sims[j, ref] = -np.inf
        nn = np.argpartition(-sims, k, axis=1)[:, :k]
        nn_sims = np.take_along_axis(sims, nn, axis=1)
        order = np.argsort(-nn_sims, axis=1)
        nn, nn_sims = np.take_along_axis(nn, order, 1), np.take_along_axis(nn_sims, order, 1)
        w = 1.0 / (1.0 - np.clip(nn_sims, -1, 0.9999) + 1e-3)
        for c in range(len(ROLES)):
            probs[s:s + 1000, c] = (w * (role_idx[nn] == c)).sum(1)
        probs[s:s + 1000] /= probs[s:s + 1000].sum(1, keepdims=True)
        top_sim[s:s + 1000] = nn_sims[:, :5].mean(1)
    return probs, top_sim


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=15)
    args = ap.parse_args()

    keys, arrays = store.load_fingerprints()
    pages = store.load_pages()
    labels = store.load_labels()
    kidx = {k: i for i, k in enumerate(keys)}
    page_ids = [tuple(k.rsplit("|", 1)) for k in keys]
    page_ids = [(f, int(p)) for f, p in page_ids]

    ref = [(fp, lab) for fp, lab in labels.items()
           if store.key(*fp) in kidx and lab["rotation"] in (0, 180) and lab["role"] in ROLES]
    print(f"{len(keys)} pages in store, {len(ref)} labelled reference pages")
    ref_rows = np.array([kidx[store.key(*fp)] for fp, _ in ref])
    ref_rot = np.array([lab["rotation"] for _, lab in ref])
    ref_roles = [lab["role"] for _, lab in ref]

    V0, V180 = page_vectors(arrays, 0), page_vectors(arrays, 180)
    R_raw = np.where((ref_rot == 0)[:, None], V0[ref_rows], V180[ref_rows])
    pca = PCA(n_components=N_COMPONENTS, random_state=0).fit(R_raw)

    def embed(X):
        Z = pca.transform(X)
        return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    R = embed(R_raw)
    row_to_ref = np.full(len(keys), -1)
    row_to_ref[ref_rows] = np.arange(len(ref_rows))
    p0, s0 = knn_votes(embed(V0), R, ref_roles, row_to_ref, args.k)
    p180, s180 = knn_votes(embed(V180), R, ref_roles, row_to_ref, args.k)

    # dominant continuation rotation per file: from labels where the file has
    # any, otherwise from continuation-looking pages whose OCR is decisive
    label_votes, ocr_votes = defaultdict(Counter), defaultdict(Counter)
    for (f, p), lab in labels.items():
        if lab["role"] == "continuation" and lab["rotation"] in (0, 180):
            label_votes[f][lab["rotation"]] += 1
    for i, (f, p) in enumerate(page_ids):
        pg = pages.get((f, p))
        looks_cont = (p0[i] if s0[i] >= s180[i] else p180[i]).argmax() == ROLES.index("continuation")
        if pg and looks_cont and pg["ocr_margin"] >= OTHER_OCR_MARGIN:
            ocr_votes[f][pg["ocr_rot"]] += 1
    file_prior = {}
    for f in set(label_votes) | set(ocr_votes):
        votes = label_votes[f] or ocr_votes[f]
        if votes:
            file_prior[f] = votes.most_common(1)[0][0]

    rows = []
    for i, (f, p) in enumerate(page_ids):
        knn_rot = 0 if s0[i] >= s180[i] else 180
        probs = p0[i] if knn_rot == 0 else p180[i]
        knn_role = ROLES[int(probs.argmax())]
        knn_conf = float(probs.max())
        pg = pages.get((f, p), {})
        ocr_rot, margin = pg.get("ocr_rot", knn_rot), pg.get("ocr_margin", 0.0)
        lab = labels.get((f, p))
        row = dict(file=f, page=p + 1, knn_role=knn_role, knn_conf=round(knn_conf, 3), knn_rot=knn_rot,
                   ocr_rot=ocr_rot, ocr_margin=margin, label_role="", label_rotation="", note="")

        if lab and lab["role"] in ROLES and lab["rotation"] in (0, 180):
            role, rotation = lab["role"], lab["rotation"]
            row.update(role_source="label", orient_source="label",
                       label_role=role, label_rotation=rotation)
            status = "ok"
            if knn_role != role and knn_conf >= CONF_OK:
                status = "role_conflict"
                row["note"] = f"neighbours say {knn_role} ({knn_conf:.0%})"
            elif margin >= CONFLICT_OCR_MARGIN and ocr_rot != rotation and \
                    (role == "front" or knn_rot == ocr_rot):
                status = "rotation_conflict"
                row["note"] = f"OCR reads it {ocr_rot} deg (margin {margin:.0%})"
        else:
            title_code = pg.get(f"title_code{ocr_rot}") or pg.get("title_code0") or pg.get("title_code180")
            if title_code and knn_role != "front":
                role, row["role_source"] = "front", "ocr"
            else:
                role, row["role_source"] = knn_role, "knn"
            orientation_doubt = False
            if role == "front":
                rotation, src = (ocr_rot, "ocr") if margin >= FRONT_OCR_MARGIN else (knn_rot, "knn")
            elif margin >= OTHER_OCR_MARGIN and ocr_rot != knn_rot:
                # measured on labelled continuations: kNN right 109x, OCR 35x
                # when they disagree - too close to call, so ask
                rotation, src = file_prior.get(f, knn_rot), "file_prior"
                orientation_doubt = True
            else:
                rotation, src = knn_rot, "knn"
            row["orient_source"] = src
            status = "ok" if (knn_conf >= CONF_OK or row["role_source"] == "ocr") else "pending"
            if orientation_doubt:
                status = "pending"
                row["note"] = f"kNN says {knn_rot} deg, OCR says {ocr_rot} deg"
        row.update(role=role, rotation=rotation, status=status,
                   title_code=pg.get(f"title_code{rotation}", ""))
        rows.append(row)

    rows.sort(key=lambda r: (r["file"], r["page"]))
    store.write_csv(store.ROLES_CSV, rows, FIELDS)
    review = [r for r in rows if r["status"] != "ok"]
    store.write_csv(store.ROLE_REVIEW_CSV, review, FIELDS)

    labelled = [r for r in rows if r["role_source"] == "label"]
    print("\nleave-one-out check of the kNN vote against the labels (rows=label, cols=kNN):")
    print("                " + "".join(f"{c:>14}" for c in ROLES))
    for a in ROLES:
        print(f"{a:>16}" + "".join(f"{sum(1 for r in labelled if r['label_role'] == a and r['knn_role'] == b):>14}"
                                   for b in ROLES))
    acc = sum(r["label_role"] == r["knn_role"] for r in labelled) / max(len(labelled), 1)
    print(f"agreement {acc:.1%}")
    for role in ROLES:
        rs = [r for r in labelled if r["label_role"] == role]
        if rs:
            ok_knn = sum(r["knn_rot"] == r["label_rotation"] for r in rs)
            dec = [r for r in rs if r["ocr_margin"] >= (FRONT_OCR_MARGIN if role == "front" else OTHER_OCR_MARGIN)]
            ok_ocr = sum(r["ocr_rot"] == r["label_rotation"] for r in dec)
            print(f"rotation vs label, {role}: kNN {ok_knn}/{len(rs)}, decisive OCR {ok_ocr}/{len(dec)}")
    print(f"\nstatus: {dict(Counter(r['status'] for r in rows))}")
    print(f"wrote {store.ROLES_CSV} and {len(review)} rows to {store.ROLE_REVIEW_CSV}")


if __name__ == "__main__":
    main()
