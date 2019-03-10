import numpy as np
import math

def simulation_Markov_chain (matrix, pi, eps):
	standard_deviation = np.inf
	steps_count = 2
	result = pi

	while (standard_deviation > eps):
		current_matrix = np.linalg.matrix_power(matrix, steps_count)
		prev_result = result
		result = np.dot(pi, current_matrix)
		steps_count += 1
		standard_deviation = calculate_standard_deviation(prev_result, result)

	return result

def calculate_standard_deviation (v1, v2):
	res = 0
	for i in range(len(v1)):
		res += (v2[i] - v1[i])**2

	return math.sqrt(res / (len(v1) - 1))

def simulation_Markov_chain_analytically (matrix, pi):
	a = matrix.transpose() - np.identity(len(matrix))
	norm = np.ones(len(matrix))
	a[len(matrix) - 1] = norm

	b = np.zeros(len(matrix))
	b[len(matrix) - 1] = 1

	return np.linalg.solve(a, b)

def main():
	matrix = np.array([[0.2, 0, 0.4, 0.1, 0, 0, 0.15, 0.15],
					[0, 0.1, 0, 0.2, 0.5, 0, 0.2, 0],
					[0.1, 0, 0.1, 0, 0, 0.4, 0.3, 0.1],
					[0, 0.1, 0, 0.3, 0, 0, 0.3, 0.3], 
					[0.4, 0, 0.2, 0.2, 0.1, 0.1, 0, 0],
					[0, 0.3, 0.15, 0, 0.4, 0.1, 0, 0.05],
					[0.1, 0.2, 0.05, 0.1, 0, 0.2, 0.05, 0.3],
					[0.2, 0.3, 0.1, 0.1, 0, 0.2, 0, 0.1]])
	
	print('Source transition probability matrix')
	print(matrix)

	eps = 0.000000001
	print('\n\nThe vectors of the distribution over the states obtained numerically with precision: ', eps)
	pi0 = [1, 0, 0, 0, 0, 0, 0, 0]
	print('\nFor initial state vector: ', pi0)
	print(simulation_Markov_chain(matrix, pi0, eps))
	pi1 = [0, 0, 1, 0, 0, 0, 0, 0]
	print('\nFor initial state vector: ', pi1)
	print(simulation_Markov_chain(matrix, pi1, eps))
	pi2 = [0.25, 0, 0.25, 0, 0.5, 0, 0, 0]
	print('\nFor initial state vector: ', pi2)
	print(simulation_Markov_chain(matrix, pi2, eps))

	print('\n\nThe vector of the distribution over the states obtained analytically')
	print(simulation_Markov_chain_analytically(matrix, pi0))
	pass

main()