# Clustering Improvements Roadmap

Three-phase plan to fix over-clustering (mega-groups of dissimilar faces) caused by DBSCAN's
chaining effect and low-quality face encodings.

| # | Change | Re-scan required? | Status |
|---|--------|-------------------|--------|
| 1 | Replace DBSCAN with HDBSCAN | No — operates on stored encodings | Done |
| 2 | Face quality filter (size-based) | No — uses stored `face_location` bounding box | Next |
| 3 | Switch `face_encodings` to `large` model | Yes — encodings are incompatible with current | Last |

## Why HDBSCAN?

Group 749 (and likely others) was a mega-cluster of unrelated faces because DBSCAN suffers from
a **chaining/bridging effect**: density-reachable transitivity means faces A→B→C all end up in
one cluster even if A and C look nothing alike. HDBSCAN uses a hierarchical minimum spanning
tree and cuts at a stable density level, which eliminates chaining and is far more conservative
about merging unrelated points.

`sklearn.cluster.HDBSCAN` is available in scikit-learn 1.5.0 — no new dependency required.

## Phase 2: Face Quality Filter

Filter out small/blurry faces before clustering using the stored `face_location` bounding box.
Small faces (low pixel area) tend to produce unreliable 128-dim encodings that pollute clusters.
This requires no video re-scan — the bounding boxes are already in the DB.

## Phase 3: Switch to `large` Encoding Model

The `face_recognition` library's default (`small`) model produces less discriminative encodings
than the `large` model. Switching eliminates a root cause of the chaining problem but requires
re-scanning all videos since the encoding vectors are incompatible between models.
