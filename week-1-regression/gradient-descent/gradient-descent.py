from numpy import *

def compute_error(b, m, data):
	total_error = 0
	
	for i in range(0, len(data)):
		x = data[i, 0]
		y = data[i, 1]
		total_error += (y - (m * x + b))**2
		
	# return average error
	return total_error / float(len(data))
	
def gradient_step(b, m, data, learning_rate):
	grad_b = 0
	grad_m = 0
	N = float(len(data))
	
	for i in range(0, len(data)):
		x = data[i, 0]
		y = data[i, 1]
		grad_b += (-2.0/N) * (y - (m * x + b))
		grad_m += (-2.0/N) * x * (y - (m * x + b))
		
	new_b = b - learning_rate * grad_b
	new_m = m - learning_rate * grad_m
	return [new_b, new_m]

def gradient_descent_runner(data, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m
	
	# gradient descent loop
	for i in range(num_iterations):
		b, m = gradient_step(b, m, data, learning_rate)
		print("Iteration ", i, ", b: ", b, ", m: ", m, ", error: ", compute_error(b, m, data))
		
	return [b, m]

def run():
	# 1. get the data
	data = genfromtxt('data.csv', delimiter=',')

	# 2. define hyperparameters of the model
	learning_rate = 0.0001
	
	## y = mx + b (slope-intercept form)
	initial_b = 0
	initial_m = 0
	num_iterations = 1000
	
	# 3. train the model
	print("Iteration progress: b: ", initial_b, ", m: ", initial_m, ", error: ", compute_error(initial_b, initial_m, data))
	[b, m] = gradient_descent_runner(data, initial_b, initial_m, learning_rate, num_iterations)
	
	print("Final state: b: ", b, ", m: ", m, ", error: ", compute_error(b, m, data))

if __name__ == '__main__':
	run()