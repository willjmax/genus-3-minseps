import pickle

with open('g3_v2_out', 'rb') as candidate_graphs:
    matrices = pickle.load(candidate_graphs)

for matrix in matrices: print matrix
