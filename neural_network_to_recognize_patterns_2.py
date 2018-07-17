from numpy import exp, array, random, dot, mean

class neural_network:
	def __init__(self, x):
		random.seed(1)

		#synapses
		self.syn0 = 2 * random.random((3,4)) - 1
		self.syn1 = 2 * random.random((4,1)) - 1

		#layers
		self.l0 = x
		self.l1 = self.__nonline(dot(self.l0, self.syn0))
		self.l2 = self.__nonline(dot(self.l1, self.syn1))

	def __nonline(self, x, deriv = False):
		if deriv == True:
			return x * (1 - x)
		
		return 1 / (1 + exp(-x))

	def train(self, x, y, num):
		for i in range(num):
			self.think(x)
			l2_error = y - self.l2

			if i % 10000 == 0:
				print("Error:" + str(mean(abs(l2_error))))

			l2_delta = l2_error * self.__nonline(self.l2, deriv = True)
			l1_error = l2_delta.dot(self.syn1.T)
			l1_delta = l1_error * self.__nonline(self.l1, deriv = True)

			#update weights
			self.syn1 += self.l1.T.dot(l2_delta)
			self.syn0 += self.l0.T.dot(l1_delta)

	def think(self, x):
		self.l0 = x
		self.l1 = self.__nonline(dot(self.l0, self.syn0))
		self.l2 = self.__nonline(dot(self.l1, self.syn1))
		return self.l2

if __name__ == "__main__":
	#input data
	x = array([[0,0,1],
			[0,1,1],
			[1,0,1],
			[1,1,1]])

	#output data
	y = array([[0],
			[1],
			[1],
			[0]])

	network = neural_network(x)
	network.train(x, y, 60000)

	print("Output after training")
	print(network.think(x))