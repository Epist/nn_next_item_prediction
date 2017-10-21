"""
Data reader

Constructs an ordinal input vector and a one-hot target vector based on a pre-ranked data dictionary.

A next item is chosen at random and the previous N items are pulled from the dictionary and assigned their relative rankings with respect to the next item.

Not sure yet how to handle cases where there are fewer than N previously purchased items. Either drop the cases or find a clean way to represent this.

Splits are handled either using online splitting or by loading a split-file that contains the row number assignments. 
Random number generator seeds will be represented explicitly and stored as a property of the reader.
"""

import numpy as np
import json
import timeit

class data_reader(object):
	def __init__(self, filename):
		self.data_fn = filename
		self.load_data(filename)


	def load_data(self, filename):
		# Load dataset and decoding dictionaries

		with open(filename, "r") as f:
			[train_dict, valid_dict, test_dict, user_id_dict, item_id_dict] = json.load(f)

		self.train_dict = train_dict
		self.valid_dict = valid_dict
		self.test_dict = test_dict
		self.user_id_dict = user_id_dict
		self.item_id_dict = item_id_dict
		self.num_users = len(self.user_id_dict.keys())
		self.num_items = len(self.item_id_dict.keys())
		print("Loaded data for ", self.num_users, " users and ", self.num_items, " items from ", filename)

	def create_user_sampling_distribution(self, data_dict, num_previous_items):
		#In order to sample uniformally from ratings without having a complicated data structure, we need to know the number of eligable items per user

		#This can be done for the training, validation, and test sets seperately

		user_ids = list(data_dict.keys())
		user_probs = np.zeros(self.num_users)
		num_eligable_ratings = 0
		num_ineligeable_users = self.num_users - len(user_ids) #Number of users not even in this train/valid/test set
		print("This set of data does not contain any items for ", num_ineligeable_users, " users.")
		for i, user in enumerate(user_ids):
			cur_user_count = len(data_dict[user]) - num_previous_items
			if cur_user_count > 0:
				user_probs[i] = cur_user_count
				num_eligable_ratings += cur_user_count
			else:
				#Print something if the user has fewer than 1 eligable item so that we know to handle this cold start scenario
				user_probs[i] = 0
				#print("User ", user, " has only ", cur_user_count + num_previous_items, " in this set and will not be utililzed")
				num_ineligeable_users += 1

		user_probs = user_probs/num_eligable_ratings

		inelibable_rate = num_ineligeable_users/self.num_users
		print(inelibable_rate, " of the users have no eligeable items in this train/val/test set of data.")

		return [user_ids, user_probs]


	def invert_id_dictionary(self, id_dictionary):
		#Function to create an inverse mapping to the original item or user ids for the purposes of identifying which items are being recommended to wihch users
		pass
		#return inverted_dictionary

	def data_gen(self, batch_size, train_valid_test, num_previous_items):
		# A generator for batches for the model.
		# A datapoint has the format [ith_item_purchased, i-1th_item_purchased, ..., user, candidate next item 1, candidate next item 2]
		# Where one of the candidate next items is the real next item that the user purchased and the other is an item drawn randomly from the set of all items \ the real next item
		# The distractor item can be an item that the user has never seen, an item they purchased previously, an item that they will purchase later, or even the current item (repeats are allowed)
		
		#The target is a number, either -1 or 1 that represents which item was the true next purchase. If the first item is the next purchase, the target is -1, otherwise it is 1.
		
		if train_valid_test == "train":
			data_dict = self.train_dict
		elif train_valid_test == "valid":
			data_dict = self.valid_dict
		elif train_valid_test == "test":
			data_dict = self.test_dict

		[user_ids, user_probs] = self.create_user_sampling_distribution(data_dict, num_previous_items)

		sun_will_rise_tomorrow = True #Assumption
		while sun_will_rise_tomorrow:
			#Pick batch_size random ratings from the train set (Do not pick ratings that are the first num_previous_items or last item in a given user's item list)

			batch_users = np.random.choice(user_ids, batch_size, p = user_probs)

			left_item_vector = np.zeros([batch_size, self.num_items])
			right_item_vector = np.zeros([batch_size, self.num_items])
			prev_items_vectors = []
			for i in range(num_previous_items):
				prev_items_vectors.append(np.zeros([batch_size, self.num_items]))
			user_vector = np.zeros([batch_size, self.num_users])
			targets = np.zeros([batch_size, 1])

			for datapoint, user in enumerate(batch_users):
				#Pick an item at random from that user
				user_item_list = data_dict[user]
				total_num_items_for_cur_user = len(user_item_list)
				#print(timeit.timeit("np.random.randint(num_previous_items, total_num_items_for_cur_user)"))

				next_item_index = np.random.randint(num_previous_items, total_num_items_for_cur_user)

				next_item = int(user_item_list[next_item_index])

				for j in range(num_previous_items): #The order here is n-1th, n-2th, n-3th, etc.
					cur_prev_item = int(user_item_list[next_item_index-j-1])
					prev_items_vectors[j][datapoint, cur_prev_item] = 1

				#Pick a random item for the contrast proportionally to the frequency of purchase (to speed up training since more frequent items are more difficult)
				while True: #To make sure it is not the identical item...
					temp_user = np.random.choice(user_ids, p = user_probs)
					distractor_item = int(np.random.choice(data_dict[temp_user]))
					if distractor_item != next_item:
						break

				user_vector[datapoint, int(user)] = 1

				#Choose a random order of presentation for the first and last item
				order = np.random.randint(2)
				if order == 0:
					left_item_vector[datapoint, next_item] = 1
					right_item_vector[datapoint, distractor_item] = 1
					targets[datapoint] = 1 #True next item is on the left
				elif order == 1:
					left_item_vector[datapoint, distractor_item] = 1
					right_item_vector[datapoint, next_item] = 1
					targets[datapoint] = -1 #True next item is on the right


			#Yield the model input-output data
			input_list = prev_items_vectors
			prev_items_vectors.extend([user_vector, left_item_vector, right_item_vector])

			yield [input_list, targets]



