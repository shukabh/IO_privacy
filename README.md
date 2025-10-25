# Input Output Privacy Record Linkage Framework

This project involves two parties: a **client** and a **server**, each possessing a dataset they are unwilling to share with one another. Despite this, the client needs to perform **record linkage** with the server's dataset and generate summary statistics from relevant columns of the server's data.

More generally this repository outlines a privacy-preserving system for performing secure, encrypted record linkage between client and server datasets while ensuring both confidentiality and statistical integrity and allowing collaborative data analysis.

## Example Scenario

Consider the following example: 

- The **client** is a researcher with a dataset containing names of HIV-infected patients from a specific county.
- The **server** is an insurance agency holding a dataset with information about all citizens in the county.

The researcher wants to create a contingency table showing the **average income** of HIV-infected patients by ethnicity. Using this framework of **homomorphic encryption**, the researcher can perform record linkage with the insurance agency's dataset and generate an **encrypted 2-way table**, which only the researcher can decrypt.

## Input Privacy
The server cannot see the dataset of the client and cannot even know how many matches are made from its dataset. 

## Output Privacy
The client cannot see the attributes of single individuals or a very small number of individuals. The results are summary statistics and should not reveal any personal information.

## Technique and threat model
Input privacy is achieved by employing homomorphic encryption. For output privacy as there is a possiblity of client being malicous to extract individual values the server devises a subquery size protocol which makes sure that client's queries if they turn out to be smaller than a threshold will be rejected by the server. 

## Project Example

In this repository, we demonstrate the framework using a sample of **5000 residents in Australia**, linking the dataset to itself and calculating the **average age** of residents by province.

## Getting Started

To get started, follow these steps:
Start a virtual enviroment

```bash
python -m venv venv && source venv/bin/activate
```

### 1. Install Dependencies  

Install the necessary packages using the following command:
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python server_sqp_f.py --server_records 10k --client_records 1k
```
### 3. Start the Client 
```bash
python client_sqp_f.py --server_records 10k --client_records 1k
```
On the server side we will see

```bash
[SERVER] Listening on localhost:1234 …
[SERVER] Client connected from ('127.0.0.1', 50329)

[SERVER] ‑‑ Record linkage: centroid pass ‑‑
[SERVER] ‑‑ Record linkage: fine scoring ‑‑
Fine scoring: 100%|█| 426/426 [08:44<0
[TIMING] Record linkage finished in 532.4s

[SERVER] ‑‑ Streaming encrypted columns for sub‑query protocol ‑‑
Streaming columns: 100%|█| 426/426 [11
[TIMING] Streaming finished in 713.8s

[SERVER] ‑‑ Δ‑protocol (sub‑query size test) ‑‑
Sending Δ ciphertexts: 100%|█| 16/16 [
    Passing keys: ['O', 'U', 'M', 'W', 'B', 'A']
    Failing keys: ['P', 'I']
[SERVER] Merged labels: ['O', 'B OR P', 'U', 'I OR M', 'W', 'A'] 

[SERVER] -- Sending final cross-tab ciphertexts (blinded) --
Sending groups: 100%|█| 6/6 [00:00<00:
[TIMING] Cross-tabulation (blinded) finished in 0.0s
[TIMING] Total processing time: 1246.2s

[SERVER] Connection closed – server done.
```

On the client side we will see
```bash
(venv) ➜  PPRL2 git:(main) ✗ python client_sqp_f.py --server_records 10k --client_records 1k
[CLIENT] Loading & preprocessing dataset …
[CLIENT] Loaded 1,000 query records
[CLIENT] Connecting to server tcp://localhost:1234 …
[CLIENT] Server reports 50 clusters.
[CLIENT] Sent public CKKS context
[CLIENT] Encrypting query signatures …
[CLIENT] Sending one-hot indicators …
One-hot: 100%|██████| 1000/1000 [00:00<00:00, 1013361.68qry/s]
[CLIENT] Max records per cluster: 426
[CLIENT] Receiving record-level score ciphertexts …
Scores: 100%|██████████████| 426/426 [08:44<00:00,  1.23s/col]
[CLIENT] Building payload mask …
[CLIENT] Uploading encrypted payload mask …
Payload ↑: 100%|███████████| 426/426 [11:51<00:00,  1.67s/col]
[CLIENT] Receiving 16 Δ-ciphertexts …
Δ ↓: 100%|████████████████| 16/16 [00:00<00:00, 1134.04ctxt/s]
[CLIENT] Receiving final averages …
Groups ↓: 100%|███████████████| 6/6 [00:00<00:00, 369.98grp/s]

[CLIENT] Final results:
| Key / Group   |   Average |
|:--------------|----------:|
| O             |   43.1754 |
| B OR P        |   47.4235 |
| U             |   44.2787 |
| I OR M        |   51.4    |
| W             |   50.9703 |
| A             |   45.5161 |
```


