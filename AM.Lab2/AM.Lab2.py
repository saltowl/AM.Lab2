import numpy as np
import math
import matplotlib.pyplot as plt

def simulation_Markov_chain (matrix, pi, eps):
	standard_deviation = np.inf
	steps_count = 2
	result = pi
	x = []
	y = []

	while (standard_deviation > eps):
		current_matrix = np.linalg.matrix_power(matrix, steps_count)
		prev_result = result
		result = np.dot(pi, current_matrix)
		standard_deviation = calculate_standard_deviation(prev_result, result)

		x.append(steps_count)
		y.append(standard_deviation)

		steps_count += 1

	plot_graph_of_standard_deviation(x, y, pi)

	return (result, steps_count - 1)

def calculate_standard_deviation (v1, v2):
	res = 0
	for i in range(len(v1)):
		res += (v2[i] - v1[i])**2

	return math.sqrt(res / (len(v1) - 1))

def simulation_Markov_chain_analytically (matrix, pi):
	a = matrix.transpose() - np.identity(len(matrix))
	a[len(matrix) - 1] = np.ones(len(matrix))

	b = np.zeros(len(matrix))
	b[len(matrix) - 1] = 1

	return np.linalg.solve(a, b)

def plot_graph_of_standard_deviation (x, y, pi):
	plt.style.use("bmh")
	plt.plot(x, y, label = 'π = (' + ', '.join([str(num) for num in pi ]) + ')')
	plt.xlabel('Number of step')
	plt.ylabel('Standard deviation between two adjacent vectors')
	plt.title('Change in standard deviation')
	plt.legend()
	pass

def print_result_of_simulation_Markov_chain(matrix, pi, eps):
	print('\nπ = ', pi)
	res = simulation_Markov_chain(matrix, pi, eps)
	print('P (', res[1], ') = ', res[0])
	pass

def main():
	transition_matrix = np.array([[0.2, 0, 0.4, 0.1, 0, 0, 0.15, 0.15],
					[0, 0.1, 0, 0.2, 0.5, 0, 0.2, 0],
					[0.1, 0, 0.1, 0, 0, 0.4, 0.3, 0.1],
					[0, 0.1, 0, 0.3, 0, 0, 0.3, 0.3], 
					[0.4, 0, 0.2, 0.2, 0.1, 0.1, 0, 0],
					[0, 0.3, 0.15, 0, 0.4, 0.1, 0, 0.05],
					[0.1, 0.2, 0.05, 0.1, 0, 0.2, 0.05, 0.3],
					[0.2, 0.3, 0.1, 0.1, 0, 0.2, 0, 0.1]])
	
	print('\nSource transition probability matrix')
	print(transition_matrix)

	eps = 0.000000001
	print('\n\nThe vectors of the distribution over the states obtained numerically with precision: ', eps)
	pi0 = [1, 0, 0, 0, 0, 0, 0, 0]
	pi1 = [0, 0, 1, 0, 0, 0, 0, 0]
	pi2 = [0.25, 0, 0.25, 0, 0.5, 0, 0, 0]
	print_result_of_simulation_Markov_chain(transition_matrix, pi0, eps)
	print_result_of_simulation_Markov_chain(transition_matrix, pi1, eps)
	print_result_of_simulation_Markov_chain(transition_matrix, pi2, eps)

	print('\n\nThe vector of the distribution over the states obtained analytically')
	print(simulation_Markov_chain_analytically(transition_matrix, pi0))

	plt.show()
	pass

main()