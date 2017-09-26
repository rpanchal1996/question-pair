import numpy as np
with open('final_clean_vectors','r') as myfile:
	vectors = myfile.readlines()
	vectors = [vector.strip() for vector in vectors]
number_of_vectors = len(vectors)
ids = np.zeros((number_of_vectors, 300), dtype='float32')
print ids.shape
final_vectors = []
for vector in vectors:
	values = vector.split()[1:]
	values = [float(value) for value in values]
	final_vectors.append(values)
final_numpy_array = np.array(final_vectors)
print final_numpy_array.shape
np.save('word_vectors',final_numpy_array)