import torch
import scipy.sparse as sp
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
from utils import csr2torch, recall_at_k, ndcg_at_k, hit_at_k, normalize_sparse_adjacency_matrix, normalize_sparse_adjacency_matrix_, filter, \
    filter_ablation_no_user, filter_ablation_no_group
from dataset import Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_directory = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,    default="CAMRa2011", # "CAMRa2011" or "Mafengwo"
    help="Either CAMRa2011 or Mafengwo",
)
parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help="Whether to print the results or not. 1 prints the results, 0 does not.",
)
parser.add_argument("--alpha", type=float, default=1, help="rate between group and user")
parser.add_argument("--power", type=float, default=1, help="For normalization of P")
parser.add_argument("--filter_pair", type=str, default="filter_1D_1D", help="pair filter of user and group")

args = parser.parse_args()
if args.verbose:
    print(f"Device: {device}")

# load dataset
dataset = args.dataset
path = current_directory + f'/data/{dataset}/'
data = Dataset(path)
R_tr_g, R_ts_g, R_tr_u, R_ts_u, C, g_neg, u_neg, gu_mat = data.getDataset()

# shape
train_n_groups = R_tr_g.shape[0]
train_group_n_items = R_tr_g.shape[1]
train_n_users = R_tr_u.shape[0]
train_user_n_items = R_tr_u.shape[1]
if args.verbose:
    print(f"number of tr_groups: {train_n_groups}")
    print(f"number of tr_groups_items: {train_group_n_items}")
    print(f"number of tr_users: {train_n_users}")
    print(f"number of tr_users_items: {train_user_n_items}")

# Graph construction
# R_tilde 구하기
new_R_tr_g = R_tr_g.to_dense()  # (group x item)
new_R_tr_u = R_tr_u.to_dense()  # (user x item)

# Augmented matrices
augmented_user_matrix = torch.cat((new_R_tr_u, gu_mat.T), dim=1)  # (user x (item + group))
augmented_group_matrix = torch.cat((new_R_tr_g, gu_mat), dim=1)  # (group x (item + user))

# Normalize the augmented matrices
augmented_user_matrix_norm = normalize_sparse_adjacency_matrix(augmented_user_matrix, 0.5)
augmented_group_matrix_norm = normalize_sparse_adjacency_matrix(augmented_group_matrix, 0.5)

# P_tilde = R^T @ R
augmented_user_P = augmented_user_matrix_norm.T @ augmented_user_matrix_norm
augmented_user_P = augmented_user_P[:train_user_n_items, :train_user_n_items]
augmented_group_P = augmented_group_matrix_norm.T @ augmented_group_matrix_norm
augmented_group_P = augmented_group_P[:train_user_n_items, :train_user_n_items]

augmented_user_P.data **= args.power
augmented_group_P.data **= args.power

new_P =  filter(augmented_user_P, augmented_group_P, args.alpha, args.filter_pair)
# new_P = filter_ablation_no_user(augmented_user_P, augmented_group_P, args.alpha, args.filter_pair) # no user info
# new_P = filter_ablation_no_group(augmented_user_P, augmented_group_P, args.alpha, args.filter_pair) # no group info

# to device
augmented_user_P = augmented_user_P.to(device=device).float()
augmented_group_P = augmented_group_P.to(device=device).float()
new_R_tr_g = new_R_tr_g.to(device=device).float()
new_R_tr_u = new_R_tr_u.to(device=device).float()
new_P = new_P.to(device=device).float()
train_user_results = new_R_tr_u @ augmented_user_P
train_group_results = new_R_tr_g @ augmented_group_P
new_results = new_R_tr_g @ new_P  # Only consider item part for final results

# Now get the results
inf_m = -99999
new_group_gt_mat = R_ts_g.to_dense()
new_results = new_results.cpu() + (inf_m) * R_tr_g.to_dense()
new_group_gt_mat = new_group_gt_mat.cpu().detach().numpy()
new_results = new_results.cpu().detach().numpy()

print(f"NEW MODEL Hit@K: {hit_at_k(new_group_gt_mat, new_results, g_neg, k=10):.4f}")
print(f"NEW MODEL NDCG@K: {ndcg_at_k(new_group_gt_mat, new_results, g_neg, k=10):.4f}")