"""
client.py  –  chunked linkage + full protocol + poly_mod override
==================================================================

Key flag: --poly_mod 32768
  Forces a larger CKKS polynomial modulus degree, which increases
  the slot count from 4,096 to 16,384. With 50 clusters this gives
  batch_size=327, so 10k queries fit in ~31 ciphertexts in ONE chunk.

  This matches the original non-batched server_sqp_f.py behavior
  that completed 100k×10k in ~9 hours.

  Without this flag, poly_mod is auto-selected (8192 for ≤50 clusters),
  requiring multiple chunks with more overhead.

Protocol: centroid → fine scoring → payload mask → Δ-protocol → cross-tab

Usage:
    # Small runs (≤1k queries): auto poly_mod is fine
    python client.py --server_records 10k --client_records 1k

    # Large runs (10k queries): force 32768 for single-chunk
    python client.py --server_records 100k --client_records 10k --poly_mod 32768

    # Custom chunk size
    python client.py --server_records 100k --client_records 10k --poly_mod 32768 --chunk_size 10000
"""

import socket
import argparse
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import tenseal as ts
from sklearn.preprocessing import StandardScaler

from helpers import *

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Chunked client: linkage + sub-query + cross-tab + evaluation"
)
parser.add_argument("--server_records", required=True, type=str)
parser.add_argument("--client_records", required=True, type=str)
parser.add_argument("--chunk_size", type=int, default=None,
                    help="Queries per chunk (auto-computed if not set)")
parser.add_argument("--poly_mod", type=int, default=None,
                    choices=[8192, 16384, 32768],
                    help="Force CKKS poly_modulus_degree (default: auto)")
parser.add_argument("--port", type=int, default=12345)
args    = parser.parse_args()
srv_rec = args.server_records
cli_rec = args.client_records
PORT    = args.port

out_dir = f"out/{srv_rec}_{cli_rec}"
os.makedirs(out_dir, exist_ok=True)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Load data                                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("[CLIENT] Loading & preprocessing dataset …")

cli_df = pd.read_pickle(
    f"dataset/{srv_rec}_{cli_rec}/"
    f"client_names_{cli_rec}_lsh200-50-100.pkl"
)
srv_df = pd.read_csv(
    f"dataset/{srv_rec}_{cli_rec}/febrl4_server_{srv_rec}.csv"
)

sig_200 = np.array(cli_df["Signature_Norm-200"].tolist())
sig_50  = np.array(cli_df["Signature_Norm-50"].tolist())

scaler = StandardScaler()
scaled_sig_200 = scaler.fit_transform(sig_200)

n_queries = sig_50.shape[0]

# Ground truth
has_gt = "ID" in cli_df.columns and "ID" in srv_df.columns
if has_gt:
    client_ids    = cli_df["ID"].values
    server_id_set = set(srv_df["ID"].values)
    true_server_id = {i: cid for i, cid in enumerate(client_ids)
                      if cid in server_id_set}
    print(f"  Ground truth: {len(true_server_id):,} / {n_queries:,} "
          f"have true matches")
else:
    print("  No ID column — evaluation will be skipped")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  Connect & handshake                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print(f"\n[CLIENT] Connecting to localhost:{PORT} …")
sock = socket.socket()
sock.connect(("localhost", PORT))

n_clusters = recv_data(sock)
print(f"  Clusters: {n_clusters}")

# ── Create CKKS context ──────────────────────────────────────────────────────
# If --poly_mod is specified, force that degree. Otherwise auto-select.
if args.poly_mod is not None:
    forced_poly = args.poly_mod
    coeff_map = {
        8192  : [60, 40, 40, 60],
        16384 : [60, 40, 40, 40, 60],
        32768 : [60, 40, 40, 40, 40, 60],
    }
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=forced_poly,
        coeff_mod_bit_sizes=coeff_map[forced_poly],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40

    BATCH_SIZE = max(1, (forced_poly // 2) // n_clusters)
    print(f"  Forced poly_mod={forced_poly}, slots={forced_poly // 2}, "
          f"batch_size={BATCH_SIZE}")
else:
    ctx = create_context(n_clusters)
    BATCH_SIZE = get_batch_size(n_clusters)
    forced_poly = choose_poly_modulus_degree(n_clusters)
    print(f"  Auto poly_mod={forced_poly}, batch_size={BATCH_SIZE}")

# ── Compute chunk_size ────────────────────────────────────────────────────────
# If not specified, pick chunk_size so all queries fit in fewest chunks
# with ~13 ciphertexts per chunk (matching the verified working profile).
# If all queries fit in one chunk, use that.
if args.chunk_size is not None:
    CHUNK = args.chunk_size
else:
    # Target: ~13 ciphertexts per chunk (same as verified 1k non-batched)
    target_ctxts = 13
    CHUNK = BATCH_SIZE * target_ctxts
    # But if all queries fit in one chunk with ≤ 50 ciphertexts, prefer that
    ctxts_for_all = (n_queries + BATCH_SIZE - 1) // BATCH_SIZE
    if ctxts_for_all <= 50:
        CHUNK = n_queries  # single chunk — fastest
    else:
        CHUNK = min(CHUNK, n_queries)

n_chunks = (n_queries + CHUNK - 1) // CHUNK
ctxts_per_chunk = (CHUNK + BATCH_SIZE - 1) // BATCH_SIZE

print(f"  chunk_size={CHUNK}, n_chunks={n_chunks}, "
      f"ciphertexts/chunk={ctxts_per_chunk}")

send_data(sock, ctx.serialize())
send_data(sock, n_queries)
send_data(sock, n_chunks)
send_data(sock, CHUNK)

max_cluster_size = recv_data(sock)
THRESHOLD        = recv_data(sock)
print(f"  max_cluster_size={max_cluster_size}, THRESHOLD={THRESHOLD}")

# Time estimate
est_ops = max_cluster_size * n_chunks * ctxts_per_chunk
print(f"\n  ── Estimated workload ──")
print(f"  Round-trips:  ~{n_chunks * 3 + max_cluster_size + 1}")
print(f"  Total ops:    {est_ops:,}")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Centroid pass
# ══════════════════════════════════════════════════════════════════════════════
print("[CLIENT] ── Phase 1: Centroid pass ──")
start_total = time.time()

best_cluster = np.zeros(n_queries, dtype=int)

for c in tqdm(range(n_chunks), desc="Centroid pass", unit="chunk"):
    c_start = c * CHUNK
    c_end   = min(c_start + CHUNK, n_queries)

    chunk_start = time.time()
    enc_scaled = ts.ckks_tensor(ctx, scaled_sig_200[c_start:c_end].tolist(),
                                None, True)
    send_data(sock, enc_scaled.serialize())
    del enc_scaled

    enc_scores = ts.ckks_tensor_from(ctx, recv_data(sock))
    scores = enc_scores.decrypt().tolist()
    del enc_scores

    for i, row in enumerate(scores):
        best_cluster[c_start + i] = int(np.argmax(row))

    if c == 0:
        t1 = time.time() - chunk_start
        print(f"\n  First chunk: {t1:.1f}s → est. total: {t1 * n_chunks:.0f}s")

centroid_time = time.time() - start_total
print(f"[TIMING] Centroid pass: {centroid_time:.1f}s")
print(f"  best_cluster[:10] = {best_cluster[:10].tolist()}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Upload chunk tensors
# ══════════════════════════════════════════════════════════════════════════════
print("[CLIENT] ── Phase 2: Uploading chunk tensors ──")
upload_start = time.time()

for c in tqdm(range(n_chunks), desc="Chunk upload", unit="chunk"):
    c_start = c * CHUNK
    c_end   = min(c_start + CHUNK, n_queries)
    c_size  = c_end - c_start

    one_hot = np.zeros((c_size, n_clusters), dtype=np.float64)
    for i in range(c_size):
        one_hot[i, best_cluster[c_start + i]] = 1.0

    enc_one_hot = ts.ckks_tensor(ctx, one_hot.tolist(), None, True)
    send_data(sock, enc_one_hot.serialize())
    del enc_one_hot

    enc_qry = ts.ckks_tensor(ctx, sig_50[c_start:c_end].tolist(), None, True)
    send_data(sock, enc_qry.serialize())
    del enc_qry

upload_time = time.time() - upload_start
print(f"[TIMING] Chunk upload: {upload_time:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: Receive fine scores
# ══════════════════════════════════════════════════════════════════════════════
print(f"[CLIENT] ── Phase 3: Fine scoring ({max_cluster_size} cols) ──")
fine_start = time.time()

all_scores = np.zeros((n_queries, max_cluster_size), dtype=np.float64)

for col_idx in tqdm(range(max_cluster_size), desc="Fine scores", unit="col"):
    bundle = recv_data(sock)

    for c in range(n_chunks):
        c_start = c * CHUNK
        c_end   = min(c_start + CHUNK, n_queries)

        enc_vec = ts.ckks_tensor_from(ctx, bundle[c])
        decrypted = enc_vec.decrypt().tolist()
        if decrypted and isinstance(decrypted[0], list):
            decrypted = [row[0] for row in decrypted]

        for i, val in enumerate(decrypted[:c_end - c_start]):
            all_scores[c_start + i, col_idx] = val

        del enc_vec

    if col_idx == 0:
        t1 = time.time() - fine_start
        est = t1 * max_cluster_size
        print(f"\n  First col: {t1:.1f}s → est. fine scoring: "
              f"{est:.0f}s ({est/3600:.1f} hrs)")

all_scores[all_scores > 1.01] = 0.0

fine_time = time.time() - fine_start
record_linkage_time = centroid_time + upload_time + fine_time
print(f"[TIMING] Fine scoring: {fine_time:.1f}s ({fine_time/3600:.1f} hrs)")
print(f"[TIMING] Record linkage total: {record_linkage_time:.1f}s "
      f"({record_linkage_time/3600:.1f} hrs)\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: Receive cluster IDs & resolve matches
# ══════════════════════════════════════════════════════════════════════════════
cluster_ids = recv_data(sock)

print("[CLIENT] ── Score distribution ──")
print(f"  min={all_scores.min():.4f}, max={all_scores.max():.4f}, "
      f"mean={all_scores.mean():.4f}")
print(f"  Scores > 0.5: {(all_scores >= 0.5).any(axis=1).sum():,} / {n_queries:,}")
print(f"  Scores > 0.8: {(all_scores >= 0.8).any(axis=1).sum():,} / {n_queries:,}")

# Argmax match resolution
print("\n[CLIENT] Building payload mask (argmax) …")

winner_ids   = []
match_scores = []
null_count   = 0

for i in range(n_queries):
    best_col  = int(np.argmax(all_scores[i]))
    candidate = cluster_ids[best_cluster[i]][best_col]
    score     = all_scores[i][best_col]

    if candidate != "NULL" and candidate is not None:
        winner_ids.append(candidate)
        match_scores.append(score)
    else:
        winner_ids.append(None)
        match_scores.append(0.0)
        null_count += 1

if null_count > 0:
    print(f"[WARNING] {null_count} queries matched NULL padding slots")

matched_ids  = [w for w in winner_ids if w is not None and w != "NULL"]
payload_mask = np.isin(cluster_ids, matched_ids).astype(int)

print(f"  Matched queries:   {len(matched_ids):,} / {n_queries:,}")
print(f"  Unique winner IDs: {len(set(matched_ids)):,}")
print(f"  Payload mask hits: {payload_mask.sum()}")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: Upload encrypted payload mask
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[CLIENT] ── Phase 5: Uploading encrypted payload mask ──")
for i in tqdm(range(max_cluster_size), desc="Payload ↑", unit="col"):
    enc_vec = ts.ckks_vector(ctx, payload_mask[:, i].tolist())
    send_data(sock, enc_vec.serialize())

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: Δ-protocol
# ══════════════════════════════════════════════════════════════════════════════
n_deltas = recv_data(sock)
print(f"\n[CLIENT] ── Phase 6: Δ-protocol ({n_deltas} ciphertexts) ──")

sign_list = []
for _ in tqdm(range(n_deltas), desc="Δ ↓", unit="ctxt"):
    enc_delta = ts.ckks_tensor_from(ctx, recv_data(sock))
    val = decrypt_scalar(enc_delta)
    sign_list.append(val)

print(f"\n  Decrypted Δ values (permuted):")
for idx, val in enumerate(sign_list):
    print(f"    delta[{idx:02d}] = {val:>12.4f}")

send_data(sock, sign_list)

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: Receive & decrypt cross-tab results
# ══════════════════════════════════════════════════════════════════════════════
print("\n[CLIENT] ── Phase 7: Receiving cross-tab results ──")
out_keys, avgs, counts = [], [], []

n_labels = recv_data(sock)
for _ in tqdm(range(n_labels), desc="Groups ↓", unit="grp"):
    lbl          = str(recv_data(sock))
    enc_num_ctxt = ts.ckks_tensor_from(ctx, recv_data(sock))
    enc_den_ctxt = ts.ckks_tensor_from(ctx, recv_data(sock))

    num = decrypt_scalar(enc_num_ctxt)
    den = decrypt_scalar(enc_den_ctxt)

    out_keys.append(lbl)
    avgs.append(round(num / den, 6) if den > 0.5 else 0.0)
    counts.append(round(den, 2))

sock.close()
total_time = time.time() - start_total

# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT: Cross-tab results
# ══════════════════════════════════════════════════════════════════════════════
df_out = pd.DataFrame({
    "Key / Group" : out_keys,
    "Count (den)" : counts,
    "Average"     : avgs,
})

print(f"\n{'=' * 70}")
print("  CROSS-TAB RESULTS")
print(f"{'=' * 70}")
print(df_out.to_markdown(index=False))

for _, row in df_out.iterrows():
    count_ok = row["Count (den)"] > 0
    avg_ok   = 0 < row["Average"] < 200
    print(f"  {row['Key / Group'][:35]:<35} "
          f"count={row['Count (den)']:>8.1f} {'✓' if count_ok else '✗ ZERO'}  "
          f"avg={row['Average']:>8.2f} {'✓' if avg_ok else '✗ RANGE'}")

df_out.to_csv(f"{out_dir}/results_full.csv", index=False)
print(f"\n  Saved → {out_dir}/results_full.csv")

# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT: Timing summary
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  TIMING SUMMARY")
print(f"{'=' * 70}")
print(f"  Centroid pass:   {centroid_time:.1f}s")
print(f"  Chunk upload:    {upload_time:.1f}s")
print(f"  Fine scoring:    {fine_time:.1f}s ({fine_time/3600:.1f} hrs)")
print(f"  Record linkage:  {record_linkage_time:.1f}s ({record_linkage_time/3600:.1f} hrs)")
print(f"  Total:           {total_time:.1f}s ({total_time/3600:.1f} hrs)")
print(f"  Queries/sec:     {n_queries / total_time:.1f}")

# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT: Linkage evaluation
# ══════════════════════════════════════════════════════════════════════════════
if has_gt:
    print(f"\n{'=' * 70}")
    print("  LINKAGE EVALUATION")
    print(f"{'=' * 70}")

    tp = fp = fn = tn = 0
    per_query = []

    for i in range(n_queries):
        cid = client_ids[i]
        pid = winner_ids[i]
        has_true = i in true_server_id
        true_id  = true_server_id.get(i, None)

        if pid is not None and has_true:
            if pid == true_id:
                tp += 1; status = "TP"
            else:
                fp += 1; fn += 1; status = "FP"
        elif pid is None and has_true:
            fn += 1; status = "FN"
        elif pid is not None and not has_true:
            fp += 1; status = "FP_no_truth"
        else:
            tn += 1; status = "TN"

        per_query.append({
            "client_id": cid,
            "predicted_server_id": pid,
            "true_server_id": true_id,
            "match_correct": (pid == true_id) if has_true else None,
            "score": match_scores[i],
            "cluster_id": int(best_cluster[i]),
            "status": status,
        })

    n_with_true = len(true_server_id)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = tp / n_with_true if n_with_true > 0 else 0.0
    reduction = 1 - (max_cluster_size /
                     (cluster_ids.shape[0] * cluster_ids.shape[1]))

    tp_scores = [match_scores[i] for i, pq in enumerate(per_query)
                 if pq["status"] == "TP"]
    fp_scores = [match_scores[i] for i, pq in enumerate(per_query)
                 if pq["status"] == "FP"]

    print(f"""
  Total queries:           {n_queries:,}
  Queries with true match: {n_with_true:,}

  ── Classification ──
  True Positives  (TP):    {tp:,}
  False Positives (FP):    {fp:,}
  False Negatives (FN):    {fn:,}
  NULL matches:            {null_count:,}

  ── Metrics ──
  Precision:               {precision:.4f}  ({precision:.1%})
  Recall:                  {recall:.4f}  ({recall:.1%})
  F1 Score:                {f1:.4f}  ({f1:.1%})
  Accuracy:                {accuracy:.4f}  ({accuracy:.1%})""")

    if tp_scores:
        print(f"""
  ── Score Analysis ──
  TP scores:  mean={np.mean(tp_scores):.4f}, min={np.min(tp_scores):.4f}, max={np.max(tp_scores):.4f}""")
    if fp_scores:
        print(f"  FP scores:  mean={np.mean(fp_scores):.4f}, "
              f"min={np.min(fp_scores):.4f}, max={np.max(fp_scores):.4f}")
    if tp_scores and fp_scores:
        print(f"  Score gap:  {np.mean(tp_scores) - np.mean(fp_scores):.4f}")

    print(f"""
  ── Efficiency ──
  Clusters:                {n_clusters}
  Max cluster size:        {max_cluster_size}
  Chunks:                  {n_chunks} × {CHUNK}
  poly_mod:                {forced_poly}
  Reduction ratio:         {reduction:.1%}
""")

    # Cluster analysis
    cluster_counts = np.bincount(best_cluster, minlength=n_clusters)
    cluster_errors = np.zeros(n_clusters, dtype=int)
    for i, pq in enumerate(per_query):
        if pq["status"] in ("FP", "FN"):
            cluster_errors[int(best_cluster[i])] += 1

    print("  ── Cluster Analysis ──")
    print(f"  Queries/cluster: min={cluster_counts.min()}, "
          f"max={cluster_counts.max()}, mean={cluster_counts.mean():.1f}")

    worst = np.argsort(cluster_errors)[::-1][:5]
    print("  Worst clusters:")
    for c in worst:
        if cluster_errors[c] == 0:
            break
        print(f"    Cluster {c}: {cluster_errors[c]} errors / "
              f"{cluster_counts[c]} queries")

    # Save outputs
    pd.DataFrame(per_query).to_csv(
        f"{out_dir}/linkage_detail.csv", index=False)
    print(f"\n  Per-query details → {out_dir}/linkage_detail.csv")

    summary = {
        "n_queries": n_queries, "n_with_true_match": n_with_true,
        "TP": tp, "FP": fp, "FN": fn, "null_matches": null_count,
        "precision": round(precision, 6), "recall": round(recall, 6),
        "f1": round(f1, 6), "accuracy": round(accuracy, 6),
        "n_clusters": n_clusters, "max_cluster_size": max_cluster_size,
        "n_chunks": n_chunks, "chunk_size": CHUNK,
        "poly_mod": forced_poly,
        "reduction_ratio": round(reduction, 4),
        "record_linkage_time_s": round(record_linkage_time, 1),
        "total_time_s": round(total_time, 1),
    }
    pd.DataFrame([summary]).to_csv(
        f"{out_dir}/linkage_summary.csv", index=False)
    print(f"  Summary metrics  → {out_dir}/linkage_summary.csv")

    errors = [pq for pq in per_query if pq["status"] in ("FP", "FN")]
    if errors:
        pd.DataFrame(errors).to_csv(
            f"{out_dir}/linkage_errors.csv", index=False)
        print(f"  Error details    → {out_dir}/linkage_errors.csv")
    else:
        print("  No errors — perfect linkage!")

print(f"\n{'=' * 70}")
print("[CLIENT] Done.")
