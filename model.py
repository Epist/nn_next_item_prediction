#Model for the siamese network version of the neural network next item prediction project


from keras.layers import Input, Dense, subtract, concatenate, Dropout
from keras.models import Model


class siamese_model(object):
	def __init__(self, num_users, num_items, num_previous_items, num_layers, num_units, activation_type, use_sparse_representation=False):
		
		self.num_users = num_users
		self.num_items = num_items
		self.num_previous_items = num_previous_items
		self.num_layers = num_layers
		self.num_units = num_units
		self.activation_type = activation_type
		self.use_sparse_representation = True

		if use_sparse_representation:
			concatenated_inputs_left = Input(shape=(self.num_items*(1+num_previous_items),), sparse=True)
			concatenated_inputs_right = Input(shape=(self.num_items*(1+num_previous_items),), sparse=True)
		else:
			past_item_input_list = []
			for i in range(num_previous_items):
				past_item_input_list.append(Input(shape=(self.num_items,)))

			left_network_item_input = Input(shape=(self.num_items,))
			right_network_item_input = Input(shape=(self.num_items,))

			#Concatenate all of the past item inputs into a single tensor
			if num_previous_items > 1:
				past_item_inputs = concatenate(past_item_input_list)
			else:
				past_item_inputs = past_item_input_list[0]

			user_input = Input(shape=(self.num_users,),)

		#Create the shared network
		shared_net = self.siamese_net_half(self.num_layers, self.num_units, self.activation_type, self.use_sparse_representation)

		#Create the left network
		if use_sparse_representation:
			left_network = shared_net.half_net(concatenated_inputs_left)
		else:
			left_network = shared_net.half_net([user_input, past_item_inputs, left_network_item_input])

		#Create the right network as a shared set of layers
		if use_sparse_representation:
			right_network = shared_net.half_net(concatenated_inputs_right)
		else:
			right_network = shared_net.half_net([user_input, past_item_inputs, right_network_item_input])

		#Merge the two halves
		output = subtract([left_network, right_network])

		#Create the model
		if use_sparse_representation:
			self.model = Model(inputs=[concatenated_inputs_left, concatenated_inputs_right], outputs=output)
		else:
			input_list = [x for x in past_item_input_list]
			input_list.extend([user_input, left_network_item_input, right_network_item_input])
			self.model = Model(inputs=input_list, outputs=output)


	def save_weights(self, filename):
		#weights = self.model.get_weights()
		self.model.save_weights(filename)

	def load_weights(self, weights):
		self.model.set_weights(weights)

	class siamese_net_half():
		#Returns a function that represents the network and can be called on a list of inputs
		def __init__(self, num_layers, num_units, activation_type, use_sparse_representation):
			self.use_sparse_representation = use_sparse_representation
			self.dense_layers = []
			for i in range(num_layers):
				self.dense_layers.append(Dense(num_units, activation=activation_type))
			self.output_layer = Dense(1, activation="sigmoid")

		def half_net(self, inputs):
			if self.use_sparse_representation:
				layer = inputs
			else:
				layer = concatenate(inputs)

			for dl in self.dense_layers:
				layer = dl(layer)

			output = self.output_layer(layer)

			return output 
