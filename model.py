import numpy as np
def load_matrices():
	q1_ids = np.load('q1_ids_matrix.npy')
	q2_ids = np.load('q2_ids_matrix.npy')
	print q1_ids[18000]
	print q2_ids[18000]
load_matrices()