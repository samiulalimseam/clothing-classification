import json
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split as surprise_train_test_split

# Load JSON data
def load_data():
    with open('../data/user_data.json', 'r') as f:
        user_data = json.load(f)

    with open('../data/product_data.json', 'r') as f:
        product_data = json.load(f)

    with open('../data/behavior_data.json', 'r') as f:
        behavior_data = json.load(f)

    with open('../data/recently_viewed_data.json', 'r') as f:
        recently_viewed_data = json.load(f)

    # Convert to pandas DataFrame
    user_df = pd.DataFrame(user_data)
    product_df = pd.DataFrame(product_data)
    behavior_df = pd.DataFrame(behavior_data)
    recently_viewed_df = pd.DataFrame(recently_viewed_data)

    return user_df, product_df, behavior_df, recently_viewed_df

user_df, product_df, behavior_df, recently_viewed_df = load_data()

# Create a Surprise dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(behavior_df[['user_id', 'product_id', 'rating']], reader)

# Split data into training and test sets
trainset, testset = surprise_train_test_split(data, test_size=0.2)

# Use the SVD algorithm
algo = SVD()

# Train the algorithm on the trainset
algo.fit(trainset)

def get_top_n_recommendations(user_id, n=5):
    # Get a list of all product_ids
    all_product_ids = product_df['product_id'].unique()

    # Get the list of products the user has already interacted with
    user_rated_products = behavior_df[behavior_df['user_id'] == user_id]['product_id'].unique()

    # Get the list of products the user has recently viewed
    user_recently_viewed_products = recently_viewed_df[recently_viewed_df['user_id'] == user_id]['product_id'].unique()

    # Combine rated and recently viewed products to exclude from recommendation
    products_to_exclude = set(user_rated_products).union(user_recently_viewed_products)

    # Filter out the products the user has already interacted with or recently viewed
    products_to_predict = [pid for pid in all_product_ids if pid not in products_to_exclude]

    # Predict the rating for each product
    predictions = [algo.predict(user_id, pid) for pid in products_to_predict]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get the top n recommendations
    top_n_recommendations = predictions[:n]

    return [pred.iid for pred in top_n_recommendations]
