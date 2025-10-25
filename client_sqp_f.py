'''
import socket
from helpers import *
import tenseal as ts
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle
import argparse

# parse shorthand sizes like "50k" → 50000
parse = lambda s: (lambda f: int(f) if f.is_integer() else f)(
    float(s[:-1]) * {'k':1e3, 'm':1e6, 'b':1e9}.get(s[-1].lower(), 1)
)

# Command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--server_records", type=str, required=True,
                    help="Number of server records, e.g., '50k'")
parser.add_argument("--client_records", type=str, required=True,
                    help="Number of client records, e.g., '1k'")
args = parser.parse_args()

server_records = args.server_records
client_records = args.client_records

# -------------------------------
# Step 0: Load and preprocess client dataset
# -------------------------------
dataset = pd.read_pickle(
    f'dataset/{server_records}_{client_records}/'
    f'client_names_{client_records}_lsh200-50-100.pkl'
)
print("Dataset loaded")

# Keep only the two signature columns
dataset = dataset[['Signature_Norm-200', 'Signature_Norm-50']]

# Normalize the 200-dimensional signatures
scaler = StandardScaler()
signatures_200 = np.array(dataset['Signature_Norm-200'].tolist())
scaled_signatures = scaler.fit_transform(signatures_200)

# Extract the full set of 50-dimensional signatures
signatures_50 = np.array(dataset['Signature_Norm-50'].tolist())

# Total number of queries = all rows
n_queries = signatures_50.shape[0]

# -------------------------------
# Step 1: Connect to server
# -------------------------------
client_socket = socket.socket()
client_socket.connect(('localhost', 1234))
print("Connected to server.")

# Receive number of clusters
no_of_clusters = recv_data(client_socket)

# -------------------------------
# Step 2: Create and send TenSEAL context
# -------------------------------
context = create_context()
send_data(client_socket, context.serialize())
print("Sent public context to server.")

# -------------------------------
# Step 3: Encrypt all query vectors and send
# -------------------------------
print("Record linkage protocol started…")
enc_querier      = ts.ckks_tensor(context, signatures_50,      None, True)
enc_querier_scaled = ts.ckks_tensor(context, scaled_signatures, None, True)
send_data(client_socket, enc_querier_scaled.serialize())

# -------------------------------
# Step 4: Receive & decrypt centroid scores
# -------------------------------
enc_res = recv_data(client_socket)
enc_sim = ts.ckks_tensor_from(context, enc_res)
cos_sims = enc_sim.decrypt().tolist()

# Find best cluster for each query
most_matching_cluster = [int(np.argmax(row)) for row in cos_sims]

# -------------------------------
# Step 5: Send one-hot cluster indicators
# -------------------------------
one_hot = np.zeros((n_queries, no_of_clusters))
for i, c in enumerate(most_matching_cluster):
    one_hot[i, c] = 1

enc_one_hot = ts.ckks_tensor(context, one_hot, None, True)
send_data(client_socket, enc_one_hot.serialize())

# -------------------------------
# Step 6: Receive max cluster size & resend raw queries
# -------------------------------
max_cluster_size = recv_data(client_socket)
send_data(client_socket, enc_querier.serialize())

# -------------------------------
# Step 7: Receive & decrypt record-level scores
# -------------------------------
res = []
for _ in range(max_cluster_size):
    enc_bytes = recv_data(client_socket)
    enc_vec   = ts.ckks_tensor_from(context, enc_bytes)
    res.append(enc_vec.decrypt().tolist())

# Shape into (n_queries, cluster_size)
res = np.array(res).T
res[res > 1.01] = 0  # threshold cleanup

print("Building payload mask…")
# -------------------------------
# Step 8: Build binary payload mask
# -------------------------------
cluster_ids   = recv_data(client_socket)
# For each query, pick the record ID with max score
id_candidates = [
    cluster_ids[ most_matching_cluster[i] ][ np.argmax(res[i]) ]
    for i in range(n_queries)
]
# mask of shape (n_records_in_cluster,)
payload_mask = np.isin(cluster_ids, id_candidates).astype(int)

# Send encrypted payload mask for each feature index
for i in range(max_cluster_size):
    enc_vec = ts.ckks_tensor(context, payload_mask[:, i])
    send_data(client_socket, enc_vec.serialize())


# -------------------------------
# Step 9: Receive and decrypt sub-query sizes
# -------------------------------
n = recv_data(client_socket)
dec_subquery_signs = []

for _ in range(n):
    vec_bytes = recv_data(client_socket)
    enc_delta = ts.ckks_tensor_from(context, vec_bytes)

    # decrypt into a PlainTensor, convert to list, then take [0]
    pt  = enc_delta.decrypt()     # PlainTensor
    lst = pt.tolist()             # e.g. [123.45]
    val = lst[0]

    #sign = int(np.sign(val))
    sign = val
    dec_subquery_signs.append(sign)

# send back the list of signs

send_data(client_socket, dec_subquery_signs)


# -------------------------------
# Step 10: Receive final ratios from server
# -------------------------------
print("Main computation started…")
keys, ratios, counts = [], [], []
num_labels = recv_data(client_socket)

for _ in range(num_labels):
    key = str(recv_data(client_socket))

    # decrypt numerator
    enc_num_ct = ts.ckks_tensor_from(context, recv_data(client_socket))
    pt_num     = enc_num_ct.decrypt()
    num        = pt_num.tolist()[0]

    # decrypt denominator
    enc_den_ct = ts.ckks_tensor_from(context, recv_data(client_socket))
    pt_den     = enc_den_ct.decrypt()
    den        = pt_den.tolist()[0]

    keys.append(key)
    counts.append(den)
    ratios.append(num/den if den != 0 else 0)

# -------------------------------
# Step 11: Output results
# -------------------------------
df = pd.DataFrame({
    'key': keys,
    'average': ratios,
    'Count': counts
})
print(df)
df.to_csv(f'out/{server_records}_{client_records}/results_full.csv', index=False)

# -------------------------------
# Step 12: Close
# -------------------------------
client_socket.close()
'''
#!/usr/bin/env python3
"""
client_subquery_protocol_with_progress.py
=========================================

Client-side driver for the privacy-preserving record-linkage demo.

Adds:
  • Rich, block-level comments so you can follow the protocol step-by-step.  
  • `tqdm` progress bars around the longest loops:
        – building the one-hot matrix
        – receiving record-level score ciphertexts
        – uploading the encrypted payload mask
        – receiving Δ-ciphertexts during the sub-query size protocol
"""

# ───────────────────────────────── Imports ──────────────────────────────────
import socket
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import tenseal as ts
from sklearn.preprocessing import StandardScaler

from helpers import *  # (de)serialisation helpers supplied in your repo

# ────────────────────────────── CLI arguments ───────────────────────────────
parser = argparse.ArgumentParser(
    description="Run client-side RL + sub-query protocol with progress bars"
)
parser.add_argument("--server_records", required=True, type=str,
                    help="e.g. '100k'  – identical to the server flag")
parser.add_argument("--client_records", required=True, type=str,
                    help="e.g. '10k'   – identical to the server flag")
args = parser.parse_args()

srv_rec = args.server_records
cli_rec = args.client_records

# ──────────────────────────── Load client data ──────────────────────────────
print("[CLIENT] Loading & preprocessing dataset …")
cli_df = pd.read_pickle(
    f"dataset/{srv_rec}_{cli_rec}/"
    f"client_names_{cli_rec}_lsh200-50-100.pkl"
)

# Keep just the two signature columns produced by your LSH pipeline
cli_df = cli_df[["Signature_Norm-200", "Signature_Norm-50"]]

# z-score normalise the 200-D signature
scaler        = StandardScaler()
sig_200       = np.array(cli_df["Signature_Norm-200"].tolist())          # (n_q × 200)
scaled_sig_200 = scaler.fit_transform(sig_200)

# 50-D signatures are already length-normalised
sig_50 = np.array(cli_df["Signature_Norm-50"].tolist())                  # (n_q × 50)

n_queries = sig_50.shape[0]
print(f"[CLIENT] Loaded {n_queries:,} query records")

# ───────────────────────────── Handshake ────────────────────────────────────
print("[CLIENT] Connecting to server tcp://localhost:1234 …")
sock = socket.socket()
sock.connect(("localhost", 12345))

# How many clusters does the server have?
n_clusters = recv_data(sock)
print(f"[CLIENT] Server reports {n_clusters} clusters.")

# ─────────────────────── TenSEAL context exchange ──────────────────────────
ctx = create_context()                       # fp = 4096, scale = 2**40 (helpers.py)
send_data(sock, ctx.serialize())
print("[CLIENT] Sent public CKKS context")

# ────────────────── Encrypt & transmit query signatures ────────────────────
print("[CLIENT] Encrypting query signatures …")
enc_q_raw    = ts.ckks_tensor(ctx, sig_50,         None, True)
enc_q_scaled = ts.ckks_tensor(ctx, scaled_sig_200, None, True)

send_data(sock, enc_q_scaled.serialize())   # → centroid pass

# ───────────────── Receive centroid scores & pick best clusters ────────────
enc_centroid_scores = ts.ckks_tensor_from(ctx, recv_data(sock))
centroid_scores     = enc_centroid_scores.decrypt().tolist()             # list(list)

best_cluster = [int(np.argmax(row)) for row in centroid_scores]          # length = n_q

# ────────────── Build one-hot (n_q × n_clusters) & send to server ──────────
print("[CLIENT] Sending one-hot indicators …")
one_hot = np.zeros((n_queries, n_clusters))

for i, c in tqdm(enumerate(best_cluster), total=n_queries,
                 desc="One-hot", unit="qry"):
    one_hot[i, c] = 1

enc_one_hot = ts.ckks_tensor(ctx, one_hot, None, True)
send_data(sock, enc_one_hot.serialize())

# ──────────────── Receive max_cluster_size & resend raw queries ────────────
max_cluster_size = recv_data(sock)
print(f"[CLIENT] Max records per cluster: {max_cluster_size}")
send_data(sock, enc_q_raw.serialize())       # → fine-scoring phase

# ─────────────── Receive encrypted record-level scores ─────────────────────
print("[CLIENT] Receiving record-level score ciphertexts …")
score_cols = []
for _ in tqdm(range(max_cluster_size), desc="Scores", unit="col"):
    enc_bytes = recv_data(sock)
    enc_vec   = ts.ckks_tensor_from(ctx, enc_bytes)
    score_cols.append(enc_vec.decrypt().tolist())        # list(list) length n_q

# Shape → (n_q, max_cluster_size)
scores = np.array(score_cols).T
scores[scores > 1.01] = 0.0                              # clip residual noise

# ─────────────── Build payload mask (one-hot among record IDs) ─────────────
cluster_ids = recv_data(sock)                            # (n_clusters × max_cluster_size)
print("[CLIENT] Building payload mask …")

# For each query choose record with highest score in its winning cluster
winner_ids = [
    cluster_ids[best_cluster[i]][ np.argmax(scores[i]) ]
    for i in range(n_queries)
]

payload_mask = np.isin(cluster_ids, winner_ids).astype(int)  # 1 if record chosen

# ─────────── Encrypt payload mask column-wise & upload to server ────────────
print("[CLIENT] Uploading encrypted payload mask …")
for i in tqdm(range(max_cluster_size), desc="Payload ↑", unit="col"):
    enc_vec = ts.ckks_tensor(ctx, payload_mask[:, i])
    send_data(sock, enc_vec.serialize())

# ────────────────── Receive Δ-ciphertexts (sub-query protocol) ──────────────
n_deltas = recv_data(sock)
print(f"[CLIENT] Receiving {n_deltas} Δ-ciphertexts …")

sign_list = []
for _ in tqdm(range(n_deltas), desc="Δ ↓", unit="ctxt"):
    enc_delta = ts.ckks_tensor_from(ctx, recv_data(sock))
    # decrypt → float, keep the raw value (server just needs its sign)
    val = enc_delta.decrypt().tolist()[0]
    sign_list.append(val)

# Send the list back (order preserved)
send_data(sock, sign_list)

# ─────────────── Receive final encrypted cross-tab results ─────────────────
print("[CLIENT] Receiving final averages …")
'''
keys, counts, avgs = [], [], []

n_labels = recv_data(sock)
for _ in tqdm(range(n_labels), desc="Groups ↓", unit="grp"):
    lbl            = str(recv_data(sock))
    enc_num_ctxt   = ts.ckks_tensor_from(ctx, recv_data(sock))
    enc_den_ctxt   = ts.ckks_tensor_from(ctx, recv_data(sock))
    num            = enc_num_ctxt.decrypt().tolist()[0]
    den            = enc_den_ctxt.decrypt().tolist()[0]
    keys.append(lbl)
    counts.append(den)
    avgs.append(num / den if den else 0.0)

df_out = pd.DataFrame({
    "Key / Group" : keys,
    "Average"     : avgs,
    "Count"       : counts,
})
'''    

keys, avgs = [], []

n_labels = recv_data(sock)
for _ in tqdm(range(n_labels), desc="Groups ↓", unit="grp"):
    lbl          = str(recv_data(sock))
    enc_num_ctxt = ts.ckks_tensor_from(ctx, recv_data(sock))
    enc_den_ctxt = ts.ckks_tensor_from(ctx, recv_data(sock))

    num = enc_num_ctxt.decrypt().tolist()[0]
    den = enc_den_ctxt.decrypt().tolist()[0]

    keys.append(lbl)
    avgs.append(num/den if den else 0.0)

# ─────────────────────────── Save / display results ────────────────────────
df_out = pd.DataFrame({
    "Key / Group" : keys,
    "Average"     : avgs
})
print("\n[CLIENT] Final results:")
print(df_out.to_markdown(index=False))

out_path = f"out/{srv_rec}_{cli_rec}/results_full.csv"
df_out.to_csv(out_path, index=False)
print(f"[CLIENT] CSV saved ➜  {out_path}")

# ─────────────────────────────── Cleanup ───────────────────────────────────
sock.close()
print("[CLIENT] Done.")
