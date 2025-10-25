import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix  # For sparse one-hot encoding
from helpers import *  # Assumes this contains kmeans_dot_product or related clustering logic
import argparse
import os

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

output_dir = f'out/{server_records}_{client_records}'
os.makedirs(output_dir, exist_ok=True)

# === Configuration ===
payload_var = 'AGE'
cross_tab_var = 'RACE'
blocking_variable = 'COUNTY'
fuzzy_names_filename = (
    f'dataset/{server_records}_{client_records}/'
    f'server_fuzzy_names_{server_records}_lsh200-50-100.pkl'
)

dname = f'server_fuzzy_names_{server_records}_lsh200-50-100'
dataset2 = pd.read_pickle(fuzzy_names_filename)

# === Step 1: Normalize the 200-dimensional fuzzy signatures ===
print("Normalizing feature vectors...")
sign_norms = dataset2["Fuzzy Signature_Norm-200"]
data_norms = np.array([x for x in sign_norms.to_numpy()])

scaler = StandardScaler()
data_norms_scaled = scaler.fit_transform(data_norms)

# === Step 2: K-means clustering ===
np.random.seed(42)
no_of_clusters = 50
name = dname + "_c" + str(no_of_clusters)
print(f"Clustering into {no_of_clusters} clusters: {name}")

cluster_centroids, cids = kmeans_dot_product(data_norms_scaled, no_of_clusters)
dataset2['Cluster_Id'] = cids

# Print cluster distribution info
count = {i: 0 for i in range(no_of_clusters)}
for i in cids:
    count[i] += 1
_min = count[min(count, key=count.get)]
_max = count[max(count, key=count.get)]
print("Cluster counts:", count)
print("Min cluster size:", _min, "Max cluster size:", _max)

# Show cluster completion over iterations
completed_items = []
for _ in range(_max):
    count = {k: v - 1 if v > 0 else 0 for k, v in count.items()}
    completed_items.append(len(dataset2) - sum(count.values()))
print("Cluster completion over max size steps:", completed_items)

# === Step 3: Save cluster histogram ===
print("Saving cluster distribution plot...")
plt.hist(cids, bins=np.arange(cids.min(), cids.max()+1, 1))
plt.title("Cluster Distribution")
plt.savefig(f"{output_dir}/{name}.eps", format='eps')
plt.close()
print("Clustering Done")

# === Step 4: Pad clusters to equal size for matrix format ===
print("Padding clusters to uniform size...")
max_cluster_size = dataset2.groupby('Cluster_Id').size().max()
print("Max cluster size:", max_cluster_size)

# Create cluster-wide feature and ID tables
columns = ['Cluster_Id'] + [f'Item_{i}' for i in range(max_cluster_size)]
cluster_dataset2 = pd.DataFrame(columns=columns)
cluster_dataset2_IDs = pd.DataFrame(columns=columns)

dummy_element = np.ones(no_of_clusters)  # Filler vector for padding

for cluster_num in range(no_of_clusters):
    cluster_items = (
        dataset2[dataset2['Cluster_Id'] == cluster_num]
               ['Fuzzy Signature_Norm-50'].tolist()
    )
    cluster_items_IDs = (
        dataset2[dataset2['Cluster_Id'] == cluster_num]['ID'].tolist()
    )
    # Pad with dummy vectors and "NULL" IDs
    while len(cluster_items) < max_cluster_size:
        cluster_items.append(dummy_element)
        cluster_items_IDs.append("NULL")

    cluster_dataset2.loc[cluster_num] = [cluster_num] + cluster_items
    cluster_dataset2_IDs.loc[cluster_num] = [cluster_num] + cluster_items_IDs

cluster_dataset2.to_pickle(f"{output_dir}/{name}.pkl")
cluster_dataset2_IDs.to_pickle(f"{output_dir}/{name}_IDs.pkl")
np.save(f"{output_dir}/{name}_centroids.npy", cluster_centroids)
print("Padding Done")
'''
# === Step 5: Map IDs to payload values ===
print("Mapping payload values to cluster matrix...")
id_to_payload = dict(zip(dataset2['ID'], dataset2[payload_var]))

safe_get_payload = lambda x: 0 if x == "NULL" else id_to_payload.get(x, 0)

payload_array = np.vectorize(safe_get_payload)(
    cluster_dataset2_IDs.drop('Cluster_Id', axis=1).to_numpy()
)

# Convert payload values to floats
to_float_or_zero = lambda v: float(v) if isinstance(v, (int, float)) or v.isnumeric() else 0.0

payload_dataset2_array = np.vectorize(to_float_or_zero)(payload_array)
payload_dataset2 = pd.DataFrame(payload_dataset2_array)
payload_dataset2.to_pickle(f"{output_dir}/{name}_payload.pkl")
print("Payload mapping done.")
'''
# === Step 5: Map IDs to payload values ===
print("Mapping payload values to cluster matrix...")
id_to_payload = dict(zip(dataset2['ID'], dataset2[payload_var]))

safe_get_payload = lambda x: 0 if x == "NULL" else id_to_payload.get(x, 0)

payload_array = np.vectorize(safe_get_payload)(
    cluster_dataset2_IDs.drop('Cluster_Id', axis=1).to_numpy()
)

# Convert payload values to floats via pandas coercion
payload_df = pd.DataFrame(payload_array)
payload_df = payload_df.apply(pd.to_numeric, errors='coerce').fillna(0)

# Save directly; payload_df is already float dtype
payload_dataset2 = payload_df
payload_dataset2.to_pickle(f"{output_dir}/{name}_payload.pkl")
print("Payload mapping done.")

# === Step 6: Map IDs to cross-tab values (e.g., race) ===
print("Mapping cross-tab values to cluster matrix...")
id_to_ct = dict(zip(dataset2['ID'], dataset2[cross_tab_var].astype(str)))
cross_tab_array = np.vectorize(id_to_ct.get)(
    cluster_dataset2_IDs.drop('Cluster_Id', axis=1).to_numpy()
)

cross_tab_dataset2 = pd.DataFrame(cross_tab_array)
cross_tab_dataset2.to_pickle(f"{output_dir}/{name}_cross_tab.pkl")
print("Cross-tab mapping done.")

# === Step 7: One-hot encode cross-tab values into a dict of sparse matrices ===
print("Creating sparse one-hot encodings for cross-tab values...")

df_dic = {}
X = cross_tab_dataset2.values  # shape: [n_clusters, max_cluster_size]
unique_values = set(v for v in X.flatten() if v != 'None')
for value in unique_values:
    mask = (X == value).astype(int)
    df_dic[value] = csr_matrix(mask)

# Pickle the sparse df_dic with the same name as before
df_dic_path = f"{output_dir}/{name}_df_dic.pkl"
with open(df_dic_path, 'wb') as f:
    pickle.dump(df_dic, f)
print("Sparse one-hot encoding saved. Script complete.")

# === Step 8: Build and save blocking_vars_server matrix ===
id_to_block = dict(zip(dataset2['ID'], dataset2[blocking_variable]))
blocking_matrix = (
    cluster_dataset2_IDs.drop('Cluster_Id', axis=1)
                      .applymap(lambda x: id_to_block.get(x, 'UNKNOWN') if x != 'NULL' else 'UNKNOWN')
)
blocking_matrix.to_pickle(
    f"{output_dir}/server_blocking_vars_matrix.pkl"
)
print("Saved blocking_vars_server to server_blocking_vars_matrix.pkl")
