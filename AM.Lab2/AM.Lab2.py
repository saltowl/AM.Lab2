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

def main():
	matrix = [[0.4, 0.6], [0.3, 0.7]]
	pi = [0.1, 0.9]
	eps = 0.1
	simulation_Markov_chain(matrix, pi, eps)
	pass

main()