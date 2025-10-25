import csv
import re
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from random import randint
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish
import itertools
from sklearn.metrics import jaccard_score
from nltk import ngrams
from sklearn.preprocessing import OneHotEncoder
import string
import hashlib
import math
from numpy.linalg import norm
import time
import psutil
import pickle
#dataframes should have FULL NAME and a FUZZY FULL NAME column respectively
import argparse

parse = lambda s: (lambda f: int(f) if f.is_integer() else f)(float(s[:-1]) * {'k':1e3, 'm':1e6, 'b':1e9}.get(s[-1].lower(), 1))


# Command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--server_records", type=str, required=True, help="Number of server records, e.g., '50k'")
parser.add_argument("--client_records", type=str, required=True, help="Number of client records, e.g., '1k'")
args = parser.parse_args()

server_records = args.server_records
client_records = args.client_records

df1 = pd.read_csv(f'dataset/{server_records}_{client_records}/ncvoter_client_{client_records}.csv')
df2 = pd.read_csv(f'dataset/{server_records}_{client_records}/ncvoter_server_{server_records}.csv')

df1['FULL NAME'] = df1['FULL NAME'].astype(str)
df2['FUZZY FULL NAME'] = df2['FUZZY FULL NAME'].astype(str)
# dtype will still show as 'object', but all values are strings

exact_names_filename = f'dataset/{server_records}_{client_records}/client_names_{client_records}_lsh200-50-100.pkl'
fuzzy_names_filename = f'dataset/{server_records}_{client_records}/server_fuzzy_names_{server_records}_lsh200-50-100.pkl'


#Parameters
shingle_size = 3
num_permutations = 200
num_permutations2 = 50
num_permutations3 = 100
max_hash = (2**20)-1 #(2 ** 31) - 1


# Function to generate the shingles of a string
def generate_shingles(string, shingle_size):
    shingles = set()
    for i in range(len(string) - shingle_size + 1):
        shingle = string[i:i + shingle_size]
        shingles.add(shingle)
    return shingles

# Function to generate a hash value for a shingle
def hash_shingle(shingle):
    return int(hashlib.sha256(shingle.encode()).hexdigest(), 32)

# Function to generate a random permutation function
def generate_permutation_function(num_permutations, max_hash):
    def permutation_function(x):
        random.seed(x)
        a = random.randint(1, max_hash)
        b = random.randint(0, max_hash)
        return lambda h: (a * h + b) % max_hash
    return [permutation_function(i) for i in range(num_permutations)]

# Function to compute the MinHash signature of a set of shingles
def compute_minhash_signature(shingles, permutation_functions):
    signature = [float('inf')] * len(permutation_functions)
    for shingle in shingles:
        shingle_hash = hash_shingle(shingle)
        for i, permutation in enumerate(permutation_functions):
            hashed_value = permutation(shingle_hash)
            if hashed_value < signature[i]:
                signature[i] = hashed_value
    return signature

permutation_functions = generate_permutation_function(num_permutations, max_hash)
permutation_functions2 = generate_permutation_function(num_permutations2, max_hash)
permutation_functions3 = generate_permutation_function(num_permutations3, max_hash)

df_names=df1

strings1 = df_names['FULL NAME']
shingles1 = [generate_shingles(string, shingle_size) for string in strings1]
#
signatures1 = [compute_minhash_signature(shingle, permutation_functions) for shingle in shingles1]
signatures12 = [compute_minhash_signature(shingle, permutation_functions2) for shingle in shingles1]
signatures13 = [compute_minhash_signature(shingle, permutation_functions3) for shingle in shingles1]
#
i=2
df_names.insert(i, 'Signature-200', signatures1)
signatures_at_responser = df_names['Signature-200'].to_numpy()
signatures_at_responser = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser])
df_names.insert(i+1, 'Signature_Norm-200', signatures_at_responser.tolist())
#
df_names.insert(i+2, 'Signature-50', signatures12)
signatures_at_responser2 = df_names['Signature-50'].to_numpy()
signatures_at_responser2 = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser2])
df_names.insert(i+3, 'Signature_Norm-50', signatures_at_responser2.tolist())
#
df_names.insert(i+4, 'Signature-100', signatures13)
signatures_at_responser3 = df_names['Signature-100'].to_numpy()
signatures_at_responser3 = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser3])
df_names.insert(i+5, 'Signature_Norm-100', signatures_at_responser3.tolist())

df_fuzzy_names=df2

strings1 = df_fuzzy_names['FUZZY FULL NAME']
shingles1 = [generate_shingles(string, shingle_size) for string in strings1]
#
signatures1 = [compute_minhash_signature(shingle, permutation_functions) for shingle in shingles1]
signatures12 = [compute_minhash_signature(shingle, permutation_functions2) for shingle in shingles1]
signatures13 = [compute_minhash_signature(shingle, permutation_functions3) for shingle in shingles1]
#
df_fuzzy_names.insert(3, 'Fuzzy Signature-200', signatures1)
signatures_at_responser = df_fuzzy_names['Fuzzy Signature-200'].to_numpy()
signatures_at_responser = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser])
df_fuzzy_names.insert(4, 'Fuzzy Signature_Norm-200', signatures_at_responser.tolist())
#
df_fuzzy_names.insert(5, 'Fuzzy Signature-50', signatures12)
signatures_at_responser2 = df_fuzzy_names['Fuzzy Signature-50'].to_numpy()
signatures_at_responser2 = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser2])
df_fuzzy_names.insert(6, 'Fuzzy Signature_Norm-50', signatures_at_responser2.tolist())
#
df_fuzzy_names.insert(7, 'Fuzzy Signature-100', signatures13)
signatures_at_responser3 = df_fuzzy_names['Fuzzy Signature-100'].to_numpy()
signatures_at_responser3 = np.array([vec/np.linalg.norm(vec) for vec in signatures_at_responser3])
df_fuzzy_names.insert(8, 'Fuzzy Signature_Norm-100', signatures_at_responser3.tolist())

df_names.to_pickle(exact_names_filename)
df_fuzzy_names.to_pickle(fuzzy_names_filename)

#print(df_names.head())
#print(df_fuzzy_names.head())