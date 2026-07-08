"""
server.py  –  chunked linkage + sub-query protocol + cross-tab
================================================================

Protocol phases:
  1. Centroid pass         (one round-trip per chunk)
  2. Chunk tensor upload   (receive all one-hot + query tensors)
  3. Fine scoring          (column-outer, chunk-inner — minimal round-trips)
  4. Send cluster IDs
  5. Receive payload mask  (one CKKSVector per column)
  6. Streaming sub-query & cross-tab accumulators
  7. Δ-protocol            (sub-query size test)
  8. Merge & send final cross-tab ciphertexts

Phases 1-4 use chunking for efficiency with large query sets.
Phases 5-8 operate on the payload mask (n_clusters × max_cluster_size),
which is independent of query count and does not need chunking.

Usage:
    python server.py --server_records 10k --client_records 10k --n_clusters 50
"""

import gc
import secrets
import socket
import time

import numpy as np
import pandas as pd
import tenseal as ts
from tqdm import tqdm

from helpers import *
import argparse

MIN_QUERY_SIZE = 5
THRESHOLD      = 0.8
SAFE_MAX       = 2 ** 15

_rng = secrets.SystemRandom()

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Chunked server: linkage + sub-query + cross-tab"
)
parser.add_argument("--server_records", type=str, required=True)
parser.add_argument("--client_records", type=str, required=True)
parser.add_argument("--n_clusters", type=int, default=50)
parser.add_argument("--port", type=int, default=12345)
args           = parser.parse_args()
server_records = args.server_records
client_records = args.client_records
n_clusters_arg = args.n_clusters
PORT           = args.port

# ── Load datasets ─────────────────────────────────────────────────────────────
print("[SERVER] Loading preprocessed datasets …")
base = (f"out/{server_records}_{client_records}/"
        f"server_fuzzy_names_{server_records}_lsh200-50-100_c{n_clusters_arg}")

dataset_paths = {
    "centroids": base + "_centroids.npy",
    "dataset2" : base + ".pkl",
    "payload"  : base + "_payload.pkl",
    "dataset3" : base + "_cross_tab.pkl",
    "df_dic"   : base + "_df_dic.pkl",
    "IDs"      : base + "_IDs.pkl",
}
datasets = {k: (pd.read_pickle(v) if v.endswith(".pkl") else np.load(v))
            for k, v in dataset_paths.items()}

cluster_centroids    = datasets["centroids"]
cluster_dataset2     = datasets["dataset2"]
payload_dataset2     = datasets["payload"]
cluster_dataset3     = cluster_dataset2.drop("Cluster_Id", axis=1)
cluster_dataset2_IDs = datasets["IDs"].drop("Cluster_Id", axis=1).to_numpy()
df_dic               = datasets["df_dic"]

max_cluster_size = cluster_dataset2.shape[1] - 1
no_of_clusters   = len(cluster_centroids)

print(f"  Clusters:          {no_of_clusters}")
print(f"  Max cluster size:  {max_cluster_size}")
print(f"  Payload shape:     {payload_dataset2.shape}")
print(f"  IDs shape:         {cluster_dataset2_IDs.shape}")
print(f"  df_dic keys:       {list(df_dic.keys())}")

# ── Pre-build TRANSPOSED column arrays ────────────────────────────────────────
print("[SERVER] Pre-building transposed column arrays …")
col_arrays_T = []
sig_dim = None

for i in range(max_cluster_size):
    col = np.array(cluster_dataset3[f"Item_{i}"].tolist()).T
    if sig_dim is None:
        sig_dim = col.shape[0]
    else:
        assert col.shape[0] == sig_dim
    col_arrays_T.append(col)

print(f"  col_arrays_T: {max_cluster_size} columns, "
      f"each ({sig_dim}, {no_of_clusters})\n")

# ── Networking ────────────────────────────────────────────────────────────────
server_socket = socket.socket()
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("localhost", PORT))
server_socket.listen(1)
print(f"[SERVER] Listening on localhost:{PORT} …")

conn, addr = server_socket.accept()
print(f"[SERVER] Client connected from {addr}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  HANDSHAKE
# ══════════════════════════════════════════════════════════════════════════════
send_data(conn, no_of_clusters)
context = ts.context_from(recv_data(conn))
print("[SERVER] Received CKKS context")

n_queries  = recv_data(conn)
n_chunks   = recv_data(conn)
chunk_size = recv_data(conn)
print(f"[SERVER] n_queries={n_queries:,}, n_chunks={n_chunks}, "
      f"chunk_size={chunk_size}")

send_data(conn, max_cluster_size)
send_data(conn, THRESHOLD)

r_max = max(2.0, SAFE_MAX / max(n_queries, 1))
print(f"[SERVER] max_cluster_size={max_cluster_size}, THRESHOLD={THRESHOLD}")
print(f"[SERVER] Δ blinding range [1.0, {r_max:.4f}]\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: Centroid pass (one round-trip per chunk)
# ══════════════════════════════════════════════════════════════════════════════
print("[SERVER] ── Phase 1: Centroid pass ──")
start_total = time.time()

for c in tqdm(range(n_chunks), desc="Centroid pass", unit="chunk"):
    enc_qry_scaled = ts.ckks_tensor_from(context, recv_data(conn))
    enc_qry_scaled.mul_(cluster_centroids)
    enc_qry_scaled.sum_(axis=2)
    send_data(conn, enc_qry_scaled.serialize())
    del enc_qry_scaled

centroid_time = time.time() - start_total
print(f"[TIMING] Centroid pass: {centroid_time:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: Receive ALL chunk tensors upfront
# ══════════════════════════════════════════════════════════════════════════════
print("[SERVER] ── Phase 2: Receiving chunk tensors ──")
upload_start = time.time()

enc_sign_chunks = []
enc_qry_chunks  = []

for c in tqdm(range(n_chunks), desc="Chunk upload", unit="chunk"):
    enc_sign = ts.ckks_tensor_from(context, recv_data(conn))
    enc_qry  = ts.ckks_tensor_from(context, recv_data(conn))
    enc_sign_chunks.append(enc_sign)
    enc_qry_chunks.append(enc_qry)

upload_time = time.time() - upload_start
print(f"[TIMING] Chunk upload: {upload_time:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3: Fine scoring (column-outer, chunk-inner)
# ══════════════════════════════════════════════════════════════════════════════
print(f"[SERVER] ── Phase 3: Fine scoring ({max_cluster_size} cols "
      f"× {n_chunks} chunks) ──")
fine_start = time.time()

for i in tqdm(range(max_cluster_size), desc="Fine scoring", unit="col"):
    col_T = col_arrays_T[i]

    chunk_results = []
    for c in range(n_chunks):
        tmp = enc_sign_chunks[c] + 0
        tmp.mul_(col_T).sum_(axis=2)
        tmp.mul_(enc_qry_chunks[c]).sum_(axis=1)
        chunk_results.append(tmp.serialize())
        del tmp

    send_data(conn, chunk_results)

fine_time = time.time() - fine_start
print(f"[TIMING] Fine scoring: {fine_time:.1f}s\n")

del enc_sign_chunks, enc_qry_chunks
gc.collect()

record_linkage_time = centroid_time + upload_time + fine_time

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4: Send cluster IDs
# ══════════════════════════════════════════════════════════════════════════════
send_data(conn, cluster_dataset2_IDs)
print("[SERVER] Sent cluster IDs")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5: Receive encrypted payload mask + streaming sub-query & cross-tab
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[SERVER] ── Phase 5: Streaming payload columns ──")
stream_start = time.time()

keys        = list(df_dic.keys())
enc_sub_acc = {k: None for k in keys}
enc_num_acc = {k: None for k in keys}

for i in tqdm(range(max_cluster_size), desc="Streaming", unit="col"):
    enc_pl = ts.ckks_vector_from(context, recv_data(conn))

    # Sub-query accumulators (denominator = count)
    for k in keys:
        mask = df_dic[k].getcol(i).toarray().ravel()
        if not np.any(mask):
            mask = mask + 1e-9
        part = enc_pl.dot(mask.tolist())
        enc_sub_acc[k] = part if enc_sub_acc[k] is None else enc_sub_acc[k] + part
        del part

    # Cross-tab accumulators (numerator = payload × mask)
    for k in keys:
        mask        = df_dic[k].getcol(i).toarray().ravel()
        payload_col = payload_dataset2.values[:, i] * mask
        if not np.any(payload_col):
            payload_col = payload_col + 1e-9
        partn = enc_pl.dot(payload_col.tolist())
        enc_num_acc[k] = partn if enc_num_acc[k] is None else enc_num_acc[k] + partn
        del partn

    del enc_pl
    gc.collect()

stream_time = time.time() - stream_start
print(f"[TIMING] Streaming: {stream_time:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 6: Δ-protocol (sub-query size test)
# ══════════════════════════════════════════════════════════════════════════════
print("[SERVER] ── Phase 6: Δ-protocol ──")

subs = []
for k in keys:
    acc = enc_sub_acc[k]
    r   = _rng.uniform(1.0, r_max)
    s   = _rng.uniform(1.0, r_max)
    subs.extend([(acc - MIN_QUERY_SIZE) * r, (MIN_QUERY_SIZE - acc) * s])

n = len(subs)
send_data(conn, n)

perm = _rng.sample(range(n), n)
invp = [0] * n
for i, p in enumerate(perm):
    invp[p] = i

for idx in tqdm(perm, desc="Δ ciphertexts", unit="ctxt"):
    send_data(conn, subs[idx].serialize())

true_list = recv_data(conn)
true_list = [true_list[i] for i in invp]

# Evaluate which keys pass
print(f"\n  {'Key':<6} {'delta1':>12} {'delta2':>12}  result")
print(f"  {'---':<6} {'-------':>12} {'-------':>12}  ------")
passing, failing = [], []
for i, k in enumerate(keys):
    d1 = true_list[2 * i]
    d2 = true_list[2 * i + 1]
    ok = d1 > 0 and d2 < 0
    print(f"  {k:<6} {d1:>12.4f} {d2:>12.4f}  {'PASS' if ok else 'FAIL'}")
    (passing if ok else failing).append(k)

print(f"\n  Passing: {passing}")
print(f"  Failing: {failing}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 7: Merge failing keys & send final cross-tab results
# ══════════════════════════════════════════════════════════════════════════════
print("[SERVER] ── Phase 7: Merge & send cross-tab ──")

if not passing:
    groups = {"ALL": set(keys)}
else:
    groups = {k: {k} for k in passing}
    for fk in failing:
        tgt = _rng.choice(passing)
        groups[tgt].add(fk)

labels        = []
enc_sub_final = {}
enc_num_final = {}
seen          = set()

for k in list(groups.keys()):
    group = groups.get(k)
    if group is None:
        continue
    tup = tuple(sorted(group))
    if tup in seen:
        continue
    seen.add(tup)
    lbl = " OR ".join(tup)
    labels.append(lbl)

    sub_acc = None
    num_acc = None
    for orig in tup:
        sub_acc = enc_sub_acc[orig] if sub_acc is None else sub_acc + enc_sub_acc[orig]
        num_acc = enc_num_acc[orig] if num_acc is None else num_acc + enc_num_acc[orig]
    enc_sub_final[lbl] = sub_acc
    enc_num_final[lbl] = num_acc

del enc_sub_acc, enc_num_acc, subs
gc.collect()

print(f"  Labels: {labels}")

cross_start = time.time()
send_data(conn, len(labels))
for lbl in tqdm(labels, desc="Sending groups", unit="grp"):
    send_data(conn, lbl)
    send_data(conn, enc_num_final[lbl].serialize())
    send_data(conn, enc_sub_final[lbl].serialize())

cross_tab_time = time.time() - cross_start

# ── Summary ───────────────────────────────────────────────────────────────────
total_time = time.time() - start_total
print(f"\n[TIMING] ── Summary ──")
print(f"  Centroid pass:  {centroid_time:.1f}s")
print(f"  Chunk upload:   {upload_time:.1f}s")
print(f"  Fine scoring:   {fine_time:.1f}s")
print(f"  Record linkage: {record_linkage_time:.1f}s")
print(f"  Streaming:      {stream_time:.1f}s")
print(f"  Cross-tab:      {cross_tab_time:.1f}s")
print(f"  Total:          {total_time:.1f}s ({total_time / 60:.1f} min)\n")

conn.close()
server_socket.close()
print("[SERVER] Done.")
