from __future__ import division
from __future__ import print_function
import pandas as pd
import numpy as np
import json



datapath = "/data1/beer/beeradvocate-crawler/ba_ratings.csv" #"/data1/movielens/ml-1m/ratings.csv" #"/data1/amazon/productGraph/categoryFiles/ratings_Video_Games.csv"#"/data1/googlelocal/reviews.clean.json"
save_filename = "data/beeradvocate/data_dicts_split_85_5_10.json"


raw_data = pd.read_csv(datapath, header=None)
raw_data.columns = ["userId", "itemId", "rating", "timestamp"]
split_ratio = [0.85,0.05,0.1]



def generate_id_dict(ratings, ID_type):
    id_dict = {}
    Ids = []    
    old_ids = list(ratings[ID_type])    
    current_id_num = 0
    for str_id in old_ids:
        if str_id not in id_dict:
            id_dict[str_id] = str(current_id_num)
            current_id_num += 1   
    if ID_type=="userId":
        print("Assigned IDs to ", current_id_num, " unique users")
    elif ID_type=="itemId":
        print("Assigned IDs to ", current_id_num, " unique items")
    return id_dict

def build_user_item_dict(data, user_id_dict, item_id_dict):
    #Group item and time of purchase data by user
    user_dict_timestamps = {}
    for i, row in raw_data.iterrows():
        raw_user = row["userId"]
        user = user_id_dict[raw_user]
        raw_item = row["itemId"]
        item = item_id_dict[raw_item]
        timestamp = str(row["timestamp"])
        if user in user_dict_timestamps:
            user_dict_timestamps[user].append((item, timestamp))
        else:
            user_dict_timestamps[user] = [(item, timestamp)]
    #Change the timestamps into ordinal rankings and sort by the ranking
    user_item_orders = {}
    for user in user_dict_timestamps:
        item_timestamp_list = user_dict_timestamps[user]
        user_item_orders[user] = sort_and_rank(item_timestamp_list)
    return user_item_orders

def sort_and_rank(item_timestamp_list):
    sorted_item_timestamp_list = sorted(item_timestamp_list, key=lambda t: int(t[1]))
    sorted_item_list = [x[0] for i,x in enumerate(sorted_item_timestamp_list)]
    return sorted_item_list
    
print("Building user dict")
user_id_dict = generate_id_dict(raw_data, "userId")
print("Building item dict")
item_id_dict = generate_id_dict(raw_data, "itemId")

print("Joining, sorting, and ranking")
user_item_dict = build_user_item_dict(raw_data, user_id_dict, item_id_dict)


def split_by_userwise_percentage(user_item_dict, train_valid_test_split):
    train_percentage = train_valid_test_split[0]
    valid_percentage = train_valid_test_split[1]
    test_percentage = train_valid_test_split[2]
    if train_percentage+valid_percentage+test_percentage != 1:
        raise(exception("Split percentages must add up to 1."))
        
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    
    for user in user_item_dict:
        purchase_list = user_item_dict[user]
        num_items = len(purchase_list)
        train_dict[user] = purchase_list[0:int(np.ceil(train_percentage*num_items))]
        valid_dict[user] = purchase_list[int(np.ceil(train_percentage*num_items)):int(np.ceil((train_percentage+valid_percentage)*num_items))]
        test_dict[user] = purchase_list[int(np.ceil((train_percentage+valid_percentage)*num_items)):]
    
    return [train_dict, valid_dict, test_dict]

print("Splitting into training, validation, and test data")
split_dicts = split_by_userwise_percentage(user_item_dict, split_ratio)

#Save split data
split_dicts.append(user_id_dict)
split_dicts.append(item_id_dict)

with open(save_filename, "w") as f:
    file_data = json.dump(split_dicts, f)

