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
        serialized = data
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


def decrypt_scalar(enc_tensor):
    """
    Safely extract a single float from a decrypted CKKS tensor.

    TenSEAL's decrypt() return type is inconsistent across tensor shapes and
    library versions — it may return any of:
      • a plain float / int       (scalar tensor)
      • a PlainTensor object      (call .tolist() to unwrap)
      • a list of floats          ([v0, v1, ...])
      • a list of lists           ([[v0], [v1], ...])

    This helper normalises all four cases to a single Python float so callers
    never need to inspect the structure themselves.

    Note on CKKS batching: when a scalar result is produced via dot() on a
    batch-encoded tensor, CKKS replicates the result across all slots.
    val[0] correctly extracts the scalar in this case.
    """
    val = enc_tensor.decrypt()

    # Unwrap PlainTensor or any ndarray-like object first
    if hasattr(val, 'tolist'):
        val = val.tolist()

    if isinstance(val, (int, float)):
        return float(val)
    if val and isinstance(val[0], (list, tuple)):
        return float(val[0][0])
    return float(val[0])


def choose_poly_modulus_degree(n_clusters):
    """
    Pick the smallest poly_modulus_degree such that the CKKS slot count
    (= poly_modulus_degree // 2) holds at least MIN_BATCH one-hot rows of
    length n_clusters.

    poly_modulus_degree | slots | coeff_mod bit-sum limit
    -------------------------------------------------------
    8192                | 4096  | 218
    16384               | 8192  | 438
    32768               | 16384 | 881
    """
    MIN_BATCH = 4

    for poly_mod in [8192, 16384, 32768]:
        slots = poly_mod // 2
        if slots // n_clusters >= MIN_BATCH:
            return poly_mod

    if (32768 // 2) // n_clusters < MIN_BATCH:
        print(f"[WARNING] n_clusters={n_clusters} is too large for "
              f"MIN_BATCH={MIN_BATCH} even at poly_mod=32768. "
              f"Effective batch_size will be 1.")
    return 32768


def create_context(n_clusters=500):
    """
    Create a TenSEAL CKKS context sized for the given number of clusters.
    poly_modulus_degree is chosen automatically via choose_poly_modulus_degree.
    """
    poly_mod = choose_poly_modulus_degree(n_clusters)

    coeff_map = {
        8192  : [60, 40, 40, 60],
        16384 : [60, 40, 40, 40, 60],
        32768 : [60, 40, 40, 40, 40, 60],
    }
    coeff_mod_bit_sizes = coeff_map[poly_mod]
    global_scale        = 2 ** 40

    slots      = poly_mod // 2
    batch_size = max(1, slots // n_clusters)
    print(f"[CONTEXT] poly_mod={poly_mod}, slots={slots}, "
          f"n_clusters={n_clusters}, batch_size={batch_size}")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    context.generate_galois_keys()
    context.global_scale = global_scale
    return context


def get_batch_size(n_clusters):
    """Return the batch size consistent with create_context for n_clusters."""
    poly_mod = choose_poly_modulus_degree(n_clusters)
    slots    = poly_mod // 2
    return max(1, slots // n_clusters)


def check_alternating_signs(lst):
    """
    Verify that each delta pair has the expected sign pattern for a passing key:
      even index (delta1) > 0  — r * (count - threshold) is positive
      odd  index (delta2) < 0  — s * (threshold - count) is negative
    Returns True only if every pair passes.
    """
    for i, val in enumerate(lst):
        if i % 2 == 0:
            if val <= 0:
                return False
        else:
            if val >= 0:
                return False
    return True


def inverse_permutation(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def writeCkks(ckks_vec, filename):
    with open(filename, 'wb') as f:
        f.write(base64.b64encode(ckks_vec))


def readCkks(filename):
    with open(filename, 'rb') as f:
        return base64.b64decode(f.read())


def kmeans_dot_product(data, k, max_iterations=20, tol=1e-4):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_
    labels    = np.zeros(len(data))

    for _ in range(max_iterations):
        distances  = np.dot(data, centroids.T)
        new_labels = np.argmax(distances, axis=1)
        if np.all(new_labels == labels):
            break
        for i in range(k):
            if np.sum(new_labels == i) > 0:
                centroids[i, :] = np.mean(data[new_labels == i, :], axis=0)
        labels = new_labels

    return centroids, labels
