#Model for the siamese network version of the neural network next item prediction project


from keras.layers import Input, Dense, subtract, concatenate, Dropout
from keras.models import Model


class model(object):
	def __init__(self, num_users, num_items, num_past_items, num_layers, num_units, activation_type):
		
		self.num_users = num_users
		self.num_items = num_items
		self.num_past_items = num_past_items
		self.num_layers = num_layers
		self.num_units = num_units
		self.activation_type = activation_type

		user_input = Input(shape=(self.num_users,))

		past_item_input_list = []
		for i in range(num_past_items):
			past_item_input_list.append(Input(shape=(self.num_items,)))

		left_network_item_input = Input(shape=(self.num_items,))
		right_network_item_input = Input(shape=(self.num_items,))

		#Concatenate all of the past item inputs into a single tensor
		past_item_inputs = concatenate()(past_item_input_list)


		#Create the shared network
		shared_net = self.siamese_net_half(self.num_layers, self.num_units, self.activation_type)

		#Create the left network
		left_network = shared_net([user_input, past_item_inputs, left_network_item_input])

		#Create the right network as a shared set of layers
		right_network = shared_net([user_input, past_item_inputs, right_network_item_input])

		#Merge the two halves
		output = subtract([left_network, right_network])

		#Create the model
		input_list = [x for x in past_item_input_list]
		input_list.extend([user_input, left_network_item_input, right_network_item_input])
		self.model = Model(inputs=input_list, outputs=output)


	def siamese_net_half(self, num_layers, num_units, activation_type):
		layer = concatenate()

		for i in range(num_layers):
			layer = Dense(num_units, activation=activation_type)(layer)

		output = Dense(1, activation="softmax")

		return output #Returns a function that represents the network and can be called on a list of inputs

	def save_weights(self, filename):
		#weights = self.model.get_weights()
		self.model.save_weights(filename)

	def load_weights(self, weights):
		self.model.set_weights(weights)