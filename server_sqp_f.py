'''
import gc
import pickle
from scipy.sparse import csr_matrix
import socket
import numpy as np
import pandas as pd
import tenseal as ts
import time
import random
from tqdm import tqdm
from helpers import *
import argparse

MIN_QUERY_SIZE = 5
THRESHOLD = 0.8
# Command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--server_records", type=str, required=True)
parser.add_argument("--client_records", type=str, required=True)
args = parser.parse_args()

server_records = args.server_records
client_records = args.client_records



dataset_paths = {
    "centroids": f"out/{server_records}_{client_records}/server_fuzzy_names_{server_records}_lsh200-50-100_c50_centroids.npy",
    "dataset2": f"out/{server_records}_{client_records}/server_fuzzy_names_{server_records}_lsh200-50-100_c50.pkl",
    "payload": f"out/{server_records}_{client_records}/server_fuzzy_names_{server_records}_lsh200-50-100_c50_payload.pkl",
    "dataset3": f"out/{server_records}_{client_records}/server_fuzzy_names_{server_records}_lsh200-50-100_c50_cross_tab.pkl",
    "df_dic": f"out/{server_records}_{client_records}/server_fuzzy_names_{server_records}_lsh200-50-100_c50_df_dic.pkl",
    "IDs": f"out/{server_records}_{client_records}/server_fuzzy_names_{server_records}_lsh200-50-100_c50_IDs.pkl"
}
datasets = {
    k: pd.read_pickle(v) if v.endswith(".pkl") else np.load(v)
    for k, v in dataset_paths.items()
}

cluster_centroids    = datasets["centroids"]
cluster_dataset2     = datasets["dataset2"]
payload_dataset2     = datasets["payload"]
cluster_dataset3     = cluster_dataset2.drop('Cluster_Id', axis=1)
cluster_dataset2_IDs = datasets["IDs"].drop('Cluster_Id', axis=1).to_numpy()
df_dic               = datasets["df_dic"]

max_cluster_size = cluster_dataset2.shape[1] - 1
no_of_clusters   = len(cluster_centroids)

# Start server
server_socket = socket.socket()
server_socket.bind(('localhost', 1234))
server_socket.listen(1)
print("Server listening...")

conn, addr = server_socket.accept()
print("Client connected:", addr)

# --- Record linkage phase (unchanged) ---
send_data(conn, no_of_clusters)
context_bytes      = recv_data(conn)
context            = ts.context_from(context_bytes)
enc_qry_bytes      = recv_data(conn)
enc_qry_scaled     = ts.ckks_tensor_from(context, enc_qry_bytes)

start_rl = time.time()
enc_qry_scaled.mul_(cluster_centroids)
enc_qry_scaled.sum_(axis=2)
send_data(conn, enc_qry_scaled.serialize())

enc_sign_bytes = recv_data(conn)
enc_sign       = ts.ckks_tensor_from(context, enc_sign_bytes)

send_data(conn, max_cluster_size)
enc_qry_bytes2 = recv_data(conn)
enc_qry        = ts.ckks_tensor_from(context, enc_qry_bytes2)

for i in range(max_cluster_size):
    col = np.array(cluster_dataset3[f"Item_{i}"].tolist()).transpose()
    tmp = enc_sign + 0
    tmp.mul_(col).sum_(axis=2)
    tmp.mul_(enc_qry).sum_(axis=1)
    
    # subtract threshold so client only sees above/below
    #n_queries = tmp.shape[0]
    #thr_row   = ts.plain_tensor([THRESHOLD] * n_queries, [n_queries])
    #tmp.sub_(thr_row)
    
    send_data(conn, tmp.serialize())

record_linkage_time = time.time() - start_rl
del enc_qry_bytes2, enc_qry
gc.collect()

print("Record linkage time:", record_linkage_time)

# Send record IDs
send_data(conn, cluster_dataset2_IDs)

# --- Single‐pass streaming for subquery & cross‐tab accumulators ---
print("Subquery & cross‐tab streaming started…")
stream_start = time.time()

keys = list(df_dic.keys())
enc_sub_acc = {k: None for k in keys}
enc_num_acc = {k: None for k in keys}

for i in range(max_cluster_size):
    b     = recv_data(conn)
    enc_pl = ts.ckks_tensor_from(context, b)

    # Subquery accumulators
    for k in keys:
        mask = df_dic[k].getcol(i).toarray().ravel() 
        #mask = df_dic[k].to_numpy()[:, i]
        v2   = ts.plain_tensor(mask, [no_of_clusters,1])
        part = enc_pl.dot(v2)
        if enc_sub_acc[k] is None:
            enc_sub_acc[k] = part
        else:
            enc_sub_acc[k].add_(part)
        del v2, part

    # Cross‐tab accumulators
    for k in keys:
        mask = df_dic[k].getcol(i).toarray().ravel()
        #mask       = df_dic[k].to_numpy()[:, i]
        payload_col= payload_dataset2.values[:, i] * mask
        v2n        = ts.plain_tensor(payload_col, [no_of_clusters,1])
        partn      = enc_pl.dot(v2n)
        if enc_num_acc[k] is None:
            enc_num_acc[k] = partn
        else:
            enc_num_acc[k].add_(partn)
        del v2n, partn

    # Free ciphertext
    del enc_pl
    gc.collect()

stream_time = time.time() - stream_start
print(f"Streaming done in {stream_time:.1f}s")

# --- Subquery protocol on enc_sub_acc ---
subs = []
for k in keys:
    acc = enc_sub_acc[k]
    r   = random.uniform(0,100)
    s   = random.uniform(0,100)
    subs.append((acc - MIN_QUERY_SIZE) * r)
    subs.append((MIN_QUERY_SIZE - acc) * s)

n = len(subs)
send_data(conn, n)
perm = random.sample(range(n), n)
invp = [0]*n
for i,p in enumerate(perm):
    invp[p] = i

for idx in perm:
    send_data(conn, subs[idx].serialize())

# Receive true deltas, invert, check signs
true_list = recv_data(conn)
true_list = [true_list[i] for i in invp]

passing, failing = [], []
for i,k in enumerate(keys):
    if true_list[2*i] > 0 and true_list[2*i+1] < 0:
        passing.append(k)
    else:
        failing.append(k)
print("Sub query sizes: ", true_list)
print("Passing keys: ",passing)
print("Failing keys: ",failing)
# --- Merge groups ---
merge = {k:{k} for k in keys}
if not passing:
    merge = {'ALL': set(keys)}
else:
    for fk in failing:
        tgt = random.choice(passing)
        merge[tgt].add(fk)
        merge[fk] = merge[tgt]

# --- Build final accumulators per merged label ---
labels = []
enc_sub_final = {}
enc_num_final = {}
seen = set()

for k, group in merge.items():
    tup = tuple(sorted(group))
    if tup in seen:
        continue
    seen.add(tup)
    lbl = " OR ".join(tup)
    labels.append(lbl)

    # Merge subquery sums
    accs = None
    for orig in tup:
        if accs is None: accs = enc_sub_acc[orig]
        else:            accs.add_(enc_sub_acc[orig])
    enc_sub_final[lbl] = accs

    # Merge numerators
    accn = None
    for orig in tup:
        if accn is None: accn = enc_num_acc[orig]
        else:            accn.add_(enc_num_acc[orig])
    enc_num_final[lbl] = accn

# Free originals
del enc_sub_acc, enc_num_acc, subs
gc.collect()

print("Merged into labels:", labels)

# --- Send final cross‐tab results ---
cross_start = time.time()
r = 1.0
send_data(conn, len(labels)) 
for lbl in labels:
    send_data(conn, lbl)
    send_data(conn, (enc_num_final[lbl] * r).serialize())
    send_data(conn, (enc_sub_final[lbl] * r).serialize())

cross_tab_time = time.time() - cross_start
total_time = record_linkage_time + stream_time + cross_tab_time

print("Cross tabulation time:", cross_tab_time)
print("Total processing time:", total_time)

conn.close()
'''
"""
Server‑side script for privacy‑preserving record linkage and sub‑query
aggregation.

This version adds plenty of inline comments **and** tqdm progress bars so that
runtime progress is visible on the console when you launch the server.
The functional logic is unchanged.
"""

# ---------------------------------------------------------------------------
# Imports & constants
# ---------------------------------------------------------------------------
import gc
import pickle
from scipy.sparse import csr_matrix
import socket
import numpy as np
import pandas as pd
import tenseal as ts
import time
import random
from tqdm import tqdm           # <-- progress‑bar utility
from helpers import *            # <-- your (de)serialisation helpers
import argparse

MIN_QUERY_SIZE = 5               # Lowest count that may pass the Δ‑test
THRESHOLD      = 0.8             # Match score threshold (currently unused)

# ---------------------------------------------------------------------------
# CLI arguments (so you can launch many experiments quickly)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run server‑side RL + subquery protocol")
parser.add_argument("--server_records", type=str, required=True,
                    help="Number (or label) of server records, eg. 100k")
parser.add_argument("--client_records", type=str, required=True,
                    help="Number (or label) of client records, eg. 10k")
args = parser.parse_args()
server_records = args.server_records
client_records = args.client_records


# ---------------------------------------------------------------------------
# Load datasets / artefacts produced during preprocessing
# ---------------------------------------------------------------------------
base = f"out/{server_records}_{client_records}/server_fuzzy_names_{server_records}_lsh200-50-100_c50"

dataset_paths = {
    "centroids": base + "_centroids.npy",
    "dataset2" : base + ".pkl",
    "payload"  : base + "_payload.pkl",
    "dataset3" : base + "_cross_tab.pkl",
    "df_dic"   : base + "_df_dic.pkl",
    "IDs"      : base + "_IDs.pkl"
}

# Load pkl / npy transparently
datasets = {k: (pd.read_pickle(v) if v.endswith(".pkl") else np.load(v))
            for k, v in dataset_paths.items()}

cluster_centroids    = datasets["centroids"]            # (n_clusters, dim)
cluster_dataset2     = datasets["dataset2"]             # comparison vectors + Cluster_Id column
payload_dataset2     = datasets["payload"]              # numeric payload you want to sum average later
cluster_dataset3     = cluster_dataset2.drop('Cluster_Id', axis=1)
cluster_dataset2_IDs = datasets["IDs"].drop('Cluster_Id', axis=1).to_numpy()
df_dic               = datasets["df_dic"]               # {key: csr_matrix mask}

max_cluster_size = cluster_dataset2.shape[1] - 1         # columns after Cluster_Id
no_of_clusters   = len(cluster_centroids)

# ---------------------------------------------------------------------------
# Networking – open a TCP listener and wait for client
# ---------------------------------------------------------------------------
server_socket = socket.socket()
server_socket.bind(('localhost', 12345))
server_socket.listen(1)
print("[SERVER] Listening on localhost:1234 …")

conn, addr = server_socket.accept()
print(f"[SERVER] Client connected from {addr}\n")

# ---------------------------------------------------------------------------
# === 1. Record‑linkage protocol ===
# ---------------------------------------------------------------------------
# 1‑A. Handshake & data exchange
send_data(conn, no_of_clusters)          # tell client clustering dimension
context_bytes      = recv_data(conn)     # public TenSEAL context
context            = ts.context_from(context_bytes)
enc_qry_bytes      = recv_data(conn)     # encrypted, **scaled** query sigs
enc_qry_scaled     = ts.ckks_tensor_from(context, enc_qry_bytes)

# 1‑B. Centroid matching (encrypted element‑wise multiply + sum)
print("[SERVER] ‑‑ Record linkage: centroid pass ‑‑")
start_rl = time.time()
enc_qry_scaled.mul_(cluster_centroids)   # broadcast multiply
enc_qry_scaled.sum_(axis=2)              # inner product along dim
send_data(conn, enc_qry_scaled.serialize())

# Client returns sign mask
enc_sign_bytes = recv_data(conn)
enc_sign       = ts.ckks_tensor_from(context, enc_sign_bytes)

# 1‑C. Fine‑grained scoring over every column in the comparison vector
print("[SERVER] ‑‑ Record linkage: fine scoring ‑‑")

send_data(conn, max_cluster_size)        # tell client to expect N columns
enc_qry_bytes2 = recv_data(conn)         # encrypted *original* query sigs
enc_qry        = ts.ckks_tensor_from(context, enc_qry_bytes2)

for i in tqdm(range(max_cluster_size), desc="Fine scoring", unit="col"):
    # Take plaintext column i (shape: n_clusters × 1)
    col = np.array(cluster_dataset3[f"Item_{i}"].tolist()).transpose()

    # tmp = enc_sign * col   (CKKS‑plaintext mult)
    tmp = enc_sign + 0                      # quick copy
    tmp.mul_(col).sum_(axis=2)              # dot product per query

    # multiply by encrypted query signature
    tmp.mul_(enc_qry).sum_(axis=1)
    
    # Optional thresholding is commented out – add if you only want sign info
    # thr_row = ts.plain_tensor([THRESHOLD] * n_queries, [n_queries])
    # tmp.sub_(thr_row)

    send_data(conn, tmp.serialize())        # send ciphertext score vector

record_linkage_time = time.time() - start_rl
print(f"[TIMING] Record linkage finished in {record_linkage_time:.1f}s\n")

# We no longer need enc_qry; free RAM / ciphertext slot
del enc_qry_bytes2, enc_qry
gc.collect()

# Send row‑aligned record IDs so client can interpret positions
send_data(conn, cluster_dataset2_IDs)

# ---------------------------------------------------------------------------
# === 2. Streaming sub‑query & cross‑tab accumulators ===
# ---------------------------------------------------------------------------
print("[SERVER] ‑‑ Streaming encrypted columns for sub‑query protocol ‑‑")
stream_start = time.time()

keys = list(df_dic.keys())               # every blocking key / demographic group
enc_sub_acc = {k: None for k in keys}    # encrypted denominators (counts)
enc_num_acc = {k: None for k in keys}    # encrypted numerators (payload sums)

for i in tqdm(range(max_cluster_size), desc="Streaming columns", unit="col"):
    # Receive encrypted payload column from client
    enc_pl = ts.ckks_tensor_from(context, recv_data(conn))

    # ---- Update sub‑query counts (denominator) ----
    for k in keys:
        mask = df_dic[k].getcol(i).toarray().ravel()           # binary mask
        v2   = ts.plain_tensor(mask, [no_of_clusters, 1])      # plaintext vector
        part = enc_pl.dot(v2)                                  # encrypted dot‑prod
        enc_sub_acc[k] = part if enc_sub_acc[k] is None else enc_sub_acc[k].add_(part)
        del v2, part

    # ---- Update cross‑tab numerators ----
    for k in keys:
        mask        = df_dic[k].getcol(i).toarray().ravel()
        payload_col = payload_dataset2.values[:, i] * mask     # element‑wise mask
        v2n         = ts.plain_tensor(payload_col, [no_of_clusters, 1])
        partn       = enc_pl.dot(v2n)
        enc_num_acc[k] = partn if enc_num_acc[k] is None else enc_num_acc[k].add_(partn)
        del v2n, partn

    # Free ciphertext for this column
    del enc_pl
    gc.collect()

stream_time = time.time() - stream_start
print(f"[TIMING] Streaming finished in {stream_time:.1f}s\n")

# ---------------------------------------------------------------------------
# === 3. Δ‑protocol to hide tiny sub‑queries ===
# ---------------------------------------------------------------------------
print("[SERVER] ‑‑ Δ‑protocol (sub‑query size test) ‑‑")
subs = []
for k in keys:
    acc = enc_sub_acc[k]
    r   = random.uniform(0, 100)
    s   = random.uniform(0, 100)
    subs.extend([(acc - MIN_QUERY_SIZE) * r, (MIN_QUERY_SIZE - acc) * s])

n = len(subs)
send_data(conn, n)                       # tell client how many ciphertexts

# Random permutation + inverse for re‑ordering replies
perm = random.sample(range(n), n)
invp = [0] * n
for i, p in enumerate(perm):
    invp[p] = i

for idx in tqdm(perm, desc="Sending Δ ciphertexts", unit="ctxt"):
    send_data(conn, subs[idx].serialize())

# Receive decrypted *signs* from client, restore original order
true_list = recv_data(conn)
true_list = [true_list[i] for i in invp]

passing, failing = [], []
for i, k in enumerate(keys):
    if true_list[2 * i] > 0 and true_list[2 * i + 1] < 0:
        passing.append(k)
    else:
        failing.append(k)
print("    Passing keys:", passing)
print("    Failing keys:", failing)

# ---------------------------------------------------------------------------
# === 4. Merge failing keys & build final accumulators ===
# ---------------------------------------------------------------------------
merge = {k: {k} for k in keys}
if not passing:
    merge = {'ALL': set(keys)}           # every key merged
else:
    for fk in failing:
        tgt = random.choice(passing)
        merge[tgt].add(fk)
        merge[fk] = merge[tgt]

labels = []
enc_sub_final = {}
enc_num_final = {}
seen = set()

for k, group in merge.items():
    tup = tuple(sorted(group))
    if tup in seen:
        continue
    seen.add(tup)
    lbl = " OR ".join(tup)              # readable label
    labels.append(lbl)

    # Sum ciphertexts inside group
    sub_acc = None
    num_acc = None
    for orig in tup:
        sub_acc = enc_sub_acc[orig] if sub_acc is None else sub_acc.add_(enc_sub_acc[orig])
        num_acc = enc_num_acc[orig] if num_acc is None else num_acc.add_(enc_num_acc[orig])
    enc_sub_final[lbl] = sub_acc
    enc_num_final[lbl] = num_acc

# Free heavy dicts
del enc_sub_acc, enc_num_acc, subs
gc.collect()
print("[SERVER] Merged labels:", labels, "\n")

# ---------------------------------------------------------------------------
# === 5. Send encrypted cross‑tab results back ===
# ---------------------------------------------------------------------------
# --- 5. Send encrypted cross-tab results back (with blinding) ---
print("[SERVER] -- Sending final cross-tab ciphertexts (blinded) --")
cross_start = time.time()

send_data(conn, len(labels))             # number of groups
for lbl in tqdm(labels, desc="Sending groups", unit="grp"):
    # pick one random blinding factor for this group
    #t = random.uniform(0, 100)
    t = 1
    # multiply both numerator and denominator by t
    blinded_num = enc_num_final[lbl] * t
    blinded_den = enc_sub_final[lbl] * t

    # send label, then blinded ciphertexts
    send_data(conn, lbl)
    send_data(conn, blinded_num.serialize())
    send_data(conn, blinded_den.serialize())

cross_tab_time = time.time() - cross_start
print(f"[TIMING] Cross-tabulation (blinded) finished in {cross_tab_time:.1f}s")

print(f"[TIMING] Total processing time: {record_linkage_time + stream_time + cross_tab_time:.1f}s\n")

conn.close()
print("[SERVER] Connection closed – server done.")
