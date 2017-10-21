"""
Recommending next item from last N items using neural networks

Represents last N in terms of ordinal rank in their item entry in the input vector. Predicts a probability distribution over next items. 

Use Python 3
"""


#from __future__ import division
#from __future__ import print_function
from data_reader import data_reader
import keras
from model import siamese_model
import pandas as pd
import numpy as np
from keras import metrics
import tensorflow as tf
import datetime
from keras.optimizers import Adagrad
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#Parameters:

#Dataset parameters 
dataset = "beeradvocate" # movielens20m, amazon_books, amazon_moviesAndTv, amazon_videoGames, amazon_clothing, beeradvocate, yelp, netflix, ml1m

#Training parameters
max_epochs = 100
batch_size = 32 
patience = 5
#val_split = [0.85, 0.05, 0.1]
early_stopping_metric = "val_binary_crossentropy" # "val_loss" #"val_accurate_RMSE"
train_epoch_length = 10000
val_epoch_length   = 1000
test_epoch_length  = 1000

#l2_weight_regulatization = None #The parameter value for the l2 weight regularization. Use None for no regularization.
#dropout_probability = None


#Model parameters
numlayers = 2
num_hidden_units = 128
num_previous_items = 1
model_save_path = "models/"
model_loss = 'binary_crossentropy' # "mean_absolute_error" 'mean_squared_error'
optimizer = 'adagrad' #Adagrad(lr=0.0025, epsilon=1e-08, decay=0.0) #'rmsprop' 'adam' 'adagrad'
activation_type = 'tanh' #Try 'selu' or 'elu' or 'softplus' or 'sigmoid'
use_sparse_representation = False #Doesn't currently work


model_save_name = "next_item_prediction_"+str(batch_size)+"bs_"+str(numlayers)+"lay_"+str(num_hidden_units)+"hu_" + str(optimizer)

#Set dataset params
if dataset == "movielens20m":
	data_path = "./data/movielens/"#'/data1/movielens/ml-20m'
elif dataset == "amazon_videoGames":
	data_path = "./data/amazon_videogames/data_dicts_split_85_5_10.json"
elif dataset == "ml1m":
	data_path = "./data/ml1m/data_dicts_split_85_5_10.json"
elif dataset == "beeradvocate":
	data_path = "./data/beeradvocate/data_dicts_split_85_5_10.json"

model_save_name += "_" + dataset + "_"
modelRunIdentifier = datetime.datetime.now().strftime("%I_%M%p_%B_%d_%Y")
model_save_name += modelRunIdentifier #Append a unique identifier to the filename

print("Loading data for " + dataset)
siamese_data_reader = data_reader(data_path)


model_handler = siamese_model(siamese_data_reader.num_users, siamese_data_reader.num_items, num_previous_items, numlayers, num_hidden_units, activation_type)
m = model_handler.model


# Grow the GPU memory useage as needed
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


m.compile(optimizer=optimizer,
              loss=model_loss,
              metrics=['mae', 'mse'])


min_loss = None
best_epoch = 0
val_history = []
for i in range(max_epochs):
	print("Starting epoch ", i+1)

	#Rebuild the generators for each epoch (the train-valid set assignments stay the same)
	train_gen = siamese_data_reader.data_gen(batch_size, "train", num_previous_items)
	valid_gen = siamese_data_reader.data_gen(batch_size, "valid", num_previous_items)

	#Train model
	#callbax = [keras.callbacks.ModelCheckpoint(model_save_path+model_save_name+"_epoch_"+str(i+1))] #Could also set save_weights_only=True
	history = m.fit_generator(train_gen, train_epoch_length, 
		validation_data=valid_gen, validation_steps=val_epoch_length) #callbacks=callbax
	
	#Early stopping code
	val_loss_list = history.history[early_stopping_metric]
	val_loss = val_loss_list[len(val_loss_list)-1]
	val_history.extend(val_loss_list)
	if min_loss == None:
		min_loss = val_loss
	elif min_loss>val_loss:
		min_loss = val_loss
		best_epoch = i
		m.save(model_save_path+model_save_name+"_epoch_"+str(i+1)+"_bestValidScore") #Only save if it is the best model (will save a lot of time and disk space...)
	elif i-best_epoch>patience:
		print("Stopping early at epoch ", i+1)
		print("Best epoch was ", best_epoch+1)
		print("Val history: ", val_history)
		break
	


#Testing
try:
	best_m = keras.models.load_model(model_save_path+model_save_name+"_epoch_"+str(best_epoch+1)+"_bestValidScore")
	best_m.save(model_save_path+model_save_name+"_bestValidScore") #resave the best one so it can be found later
	test_epoch = best_epoch+1
except:
	print("FAILED TO LOAD BEST MODEL. TESTING WITH MOST RECENT MODEL.")
	best_m = m
	test_epoch = i+1

print("Testing model from epoch: ", test_epoch)


print("\nEvaluating model with fixed split")
test_gen = siamese_data_reader.data_gen(batch_size, "test", num_previous_items)
test_results = best_m.evaluate_generator(test_gen, test_epoch_length)
print("Test results with fixed split")
#print(test_results)
for i in range(len(test_results)):
	print(m.metrics_names[i], " : ", test_results[i])

"""
print("Testing manually")
test_gen_manual = data_reader.data_gen(batch_size, None, train_val_test = "test", shuffle=shuffle_data_every_epoch, auxilliary_mask_type = auxilliary_mask_type, aux_var_value = aux_var_value, return_target_count=True)

predictions = []
targets = []
ratings_count = 0
print("Predicting")
for i in range(int(np.floor(data_reader.test_set_size/batch_size))):
	current_data = test_gen_manual.next()
	input_list = current_data[0]
	current_targets = current_data[1]
	cur_ratings_count = current_data[2]
	targets.append(current_targets)
	ratings_count += cur_ratings_count
	current_preds = best_m.predict(input_list, batch_size=batch_size, verbose=0)
	predictions.append(current_preds)

print("Computing error")
def compute_full_RMSE(predictions, targets, ratings_count):
	sum_squared_error = 0
	for i in range(len(predictions)):
		cur_preds = predictions[i]
		cur_tars = targets[i]
		error_contribution = np.sum(np.square(np.subtract(cur_preds, cur_tars)))
		sum_squared_error += error_contribution
	MSE = sum_squared_error/ratings_count
	RMSE = np.sqrt(MSE)
	return RMSE

RMSE = compute_full_RMSE(predictions, targets, ratings_count)
print("Manual test RMSE is ", RMSE)
"""