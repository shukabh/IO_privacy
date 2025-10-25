import pickle
import struct
import tenseal as ts
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import base64
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm
import gc
import time

def send_data(sock, data, raw=False):
    if not raw:
        serialized = pickle.dumps(data)
    else:
        serialized = data  # Already bytes, e.g., from .serialize()

    sock.sendall(struct.pack('>I', len(serialized)) + serialized)

def recv_data(sock, raw=False):
    def recvall(n):
        buf = b''
        while len(buf) < n:
            part = sock.recv(n - len(buf))
            if not part:
                raise ConnectionError("Socket connection closed")
            buf += part
        return buf

    raw_length = recvall(4)
    msg_length = struct.unpack('>I', raw_length)[0]
    data = recvall(msg_length)

    return data if raw else pickle.loads(data)


def check_alternating_signs(lst):
    for i, val in enumerate(lst):
        if i % 2 == 0:
            if val <= 0:
                return False
        else:
            if val >= 0:
                return False
    return True


def inverse_permutation(perm):
    # Create an array to store the inverse
    inverse = [0] * len(perm)
    
    # Fill the inverse permutation
    for i, p in enumerate(perm):
        inverse[p] = i
    
    return inverse

def writeCkks(ckks_vec, filename):
    ser_ckks_vec = base64.b64encode(ckks_vec)

    with open(filename, 'wb') as f:
        f.write(ser_ckks_vec)

def readCkks(filename):
    with open(filename, 'rb') as f:
        ser_ckks_vec = f.read()
    
    return base64.b64decode(ser_ckks_vec)

def create_context():
    poly_modulus_degree = 2*4096
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    global_scale= 2**40

    context = ts.context(
                ts.SCHEME_TYPE.CKKS, 
                poly_modulus_degree = poly_modulus_degree,
                coeff_mod_bit_sizes = coeff_mod_bit_sizes
                )
    context.generate_galois_keys()
    context.global_scale = global_scale

    return context

def kmeans_dot_product(data, k, max_iterations=20, tol=1e-4):
    #centroids = data[np.random.choice(range(len(data)), k, replace=False)]
    
    # Get cluster centroids (Slower but more precise)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    
    # Initialize centroids randomly (it runs faster, both results are little different.)
    #centroids = data[np.random.choice(len(data), k, replace=False), :]
    
    labels = np.zeros(len(data))

    for index in range(max_iterations):
        #print(index)
        # Assign each data point to the nearest centroid using dot product
        distances = np.dot(data, centroids.T)
        new_labels = np.argmax(distances, axis=1)
        #print(new_labels)

        # Check for convergence
        if np.all(new_labels == labels):
            break

        # Update centroids
        for i in range(k):
            if np.sum(new_labels == i) > 0:
                centroids[i, :] = np.mean(data[new_labels == i, :], axis=0)
                #print(i, "update")

        labels = new_labels

    return centroids, labels

def subquery_protocol_chunked(conn, context, df_dic, enc_payload_querier, max_cluster_size, no_of_clusters, min_query_size=5, chunk_size=32):
    sub_query_protocol = []
    sub_query_sizes = {}

    for key in tqdm(df_dic.keys(), desc="Subquery Protocol Progress"):
        sub_query_size = 0
        for i in range(max_cluster_size):
            #v2 = ts.plain_tensor(df_dic[key].to_numpy()[:, i], [no_of_clusters, 1])
            enc_payload_querier_bytes = recv_data(conn)
            payload_querier = ts.ckks_tensor_from(context, enc_payload_querier_bytes)  
            
            mat = df_dic[key]
            col_vector = mat[:, i].toarray().reshape(-1)  # works for sparse
            v2 = ts.plain_tensor(col_vector, [no_of_clusters, 1])
            
            sub_query_size += payload_querier.dot(v2)
            del v2

        r = random.uniform(0, 100)
        s = random.uniform(0, 100)
        delta_query1 = r * (sub_query_size - min_query_size)
        delta_query2 = s * (min_query_size - sub_query_size)

        sub_query_protocol.append(delta_query1)
        sub_query_protocol.append(delta_query2)
        sub_query_sizes[key] = sub_query_size

        del sub_query_size, delta_query1, delta_query2
        gc.collect()

    # Step 2: Permute
    n = len(sub_query_protocol)
    permutation = random.sample(range(n), n)
    inverse_perm = [0]*n
    for i, p in enumerate(permutation):
        inverse_perm[p] = i

    sub_query_sizes_perm = [sub_query_protocol[i] for i in permutation]

    # Step 3: Send size
    send_data(conn, n)

    # Step 4: Chunked send
    for i in range(0, n, chunk_size):
        chunk = [x.serialize() for x in sub_query_sizes_perm[i:i+chunk_size]]
        send_data(conn, pickle.dumps(chunk))

    return sub_query_sizes, permutation, inverse_perm


def subquery_protocol_sparser(conn, context, df_dic,
                             max_cluster_size, no_of_clusters,
                             min_query_size=5, chunk_size=32):
    # Initialize accumulators
    sub_query_sizes = { key: 0 for key in df_dic.keys() }
    keys = list(df_dic.keys())

    # Step A: For each column i, receive one ciphertext and update every key’s running sum
    for i in tqdm(range(max_cluster_size), desc="Receiving & Processing Columns"):
        # 1) Receive the encrypted mask for column i
        enc_bytes = recv_data(conn)
        enc_vec   = ts.ckks_tensor_from(context, enc_bytes)

        # 2) For each key, build plaintext for column i and dot
        for key in keys:
            # Avoid full toarray on the entire matrix: slice column only
            col_sparse = df_dic[key].getcol(i)           # still CSR
            nz_idx     = col_sparse.indices              # non-zero row indices
            # Build a small dense vector only at nz positions
            arr        = [0.0] * no_of_clusters
            for idx_val, val in zip(nz_idx, col_sparse.data):
                arr[idx_val] = val
            plain_vec  = ts.plain_tensor(arr, [no_of_clusters, 1])

            # Homomorphic dot and accumulate
            sub_query_sizes[key] += enc_vec.dot(plain_vec)

            # Free plaintext tensor
            del plain_vec

        # 3) Drop ciphertext & force cleanup before next iteration
        del enc_vec
        gc.collect()

    # Step B: Build the two‐delta protocol list
    protocol_list = []
    for key in keys:
        sz = sub_query_sizes[key]
        #r, s = random.random(), random.random()
        r, s = 1, 1
        protocol_list += [
            r * (sz - min_query_size),
            s * (min_query_size - sz)
        ]

    # Step C: Permute and send counts back
    n = len(protocol_list)
    permutation   = random.sample(range(n), n)
    inverse_perm  = [0]*n
    for i, p in enumerate(permutation):
        inverse_perm[p] = i
    permuted_list = [protocol_list[i] for i in permutation]

    # 1) Send total count
    send_data(conn, n)

    # 2) Send in chunks
    for start in range(0, n, chunk_size):
        chunk_bytes = pickle.dumps(
            [x.serialize() for x in permuted_list[start:start+chunk_size]]
        )
        send_data(conn, chunk_bytes)

    return inverse_perm


def subquery_protocol_sparse(conn, context, df_dic, enc_payload_querier, max_cluster_size, no_of_clusters, min_query_size=5, chunk_size=32):
    sub_query_protocol = []
    sub_query_sizes = {}

    for key in tqdm(df_dic.keys(), desc="Subquery Protocol Progress"):
        sub_query_size = 0
        for i in range(max_cluster_size):
            #v2 = ts.plain_tensor(df_dic[key].to_numpy()[:, i], [no_of_clusters, 1])
            mat = df_dic[key]
            col_vector = mat[:, i].toarray().reshape(-1)  # works for sparse
            v2 = ts.plain_tensor(col_vector, [no_of_clusters, 1])
            
            sub_query_size += enc_payload_querier[i].dot(v2)
            del v2

        r = random.uniform(0, 100)
        s = random.uniform(0, 100)
        delta_query1 = r * (sub_query_size - min_query_size)
        delta_query2 = s * (min_query_size - sub_query_size)

        sub_query_protocol.append(delta_query1)
        sub_query_protocol.append(delta_query2)
        sub_query_sizes[key] = sub_query_size

        del sub_query_size, delta_query1, delta_query2
        gc.collect()

    # Step 2: Permute
    n = len(sub_query_protocol)
    permutation = random.sample(range(n), n)
    inverse_perm = [0]*n
    for i, p in enumerate(permutation):
        inverse_perm[p] = i

    sub_query_sizes_perm = [sub_query_protocol[i] for i in permutation]

    # Step 3: Send size
    send_data(conn, n)

    # Step 4: Chunked send
    for i in range(0, n, chunk_size):
        chunk = [x.serialize() for x in sub_query_sizes_perm[i:i+chunk_size]]
        send_data(conn, pickle.dumps(chunk))

    return sub_query_sizes, permutation, inverse_perm


def check_alternating_signs(lst):
    # Expect pairs of values: first <0, second >0 for pass
    for i in range(0, len(lst), 2):
        if not (lst[i] < 0 and lst[i+1] > 0):
            return False
    return True
