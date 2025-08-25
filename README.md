# Zomato Restaurant Recommendation System


This project develops a restaurant recommendation system using the Zomato Bengaluru dataset. The goal is to provide personalized restaurant suggestions to users based on various factors, including restaurant attributes, user preferences, and potentially a combination of content-based and collaborative filtering techniques.

## Project Overview

The project involves the following key stages:

1.  **Data Loading and Exploration**: Loading the Zomato dataset and performing initial exploration to understand its structure, content, and identify missing values.
2.  **Exploratory Data Analysis (EDA)**: Visualizing key patterns and insights from the dataset, such as:
    *   Top restaurant chains and their distribution.
    *   Popular cuisines in Bengaluru.
    *   Distribution of online ordering and table booking availability.
    *   Distribution of restaurant ratings.
    *   Most common restaurant types.
    *   Distribution of approximate cost for two people.
    *   Location-based patterns of restaurants.
    *   Relationships between numerical variables like votes, ratings, and cost.
3.  **Feature Engineering for Recommendation**: Preparing the data for building a recommendation system by:
    *   Selecting relevant columns for recommendation.
    *   Handling missing values in the selected columns.
    *   Converting data types for numerical features.
    *   Cleaning and standardizing text data.
    *   Extracting features like the number of cuisines.
    *   Applying TF-IDF vectorization to text features ('cuisines' and 'dish\_liked').
    *   Scaling numerical features and encoding categorical features.
    *   Creating a combined feature matrix for similarity calculations.
4.  **Similarity Calculation**: Calculating similarity scores between restaurants using cosine similarity based on the engineered features. This similarity matrix is a core component for a content-based or hybrid recommendation system.
5.  **Analyzing Reviews**: Extracting and analyzing restaurant reviews and ratings to understand the distribution of ratings from customer feedback.
6.  **Wordcloud of Dishes Liked**: Generating wordclouds to visualize the most liked dishes for different cuisines, providing insights into popular food items.

## Dataset

The dataset used in this project is `zomato.csv`, containing information about restaurants in Bengaluru, India.

## Key Findings from EDA

*   "San Churro Cafe" and "New Prashanth Hotel" are among the top restaurant chains based on the number of outlets.
*   North Indian, Chinese, and South Indian are the most popular cuisines.
*   A majority of restaurants offer online ordering, while a smaller proportion provide table booking.
*   Restaurant ratings are generally high, with a peak around 3.8 to 4.0.
*   "Quick Bites" and "Casual Dining" are the most common restaurant types.
*   Most restaurants have an approximate cost for two people in the lower to mid-ranges.
*   Restaurants are clustered in certain areas of Bengaluru (e.g., Banashankari, Basavanagudi).
*   There is a positive correlation between votes and ratings, and a weak positive correlation between cost and ratings.

## Feature Engineering Details

The following features were engineered for the recommendation system:

*   **Numerical Features**: Scaled 'rate', 'votes', 'approx\_cost(for two people)', and 'num\_cuisines'.
*   **Categorical Features**: One-Hot Encoded 'location' and 'rest\_type'.
*   **Text Features**: TF-IDF vectors for 'cuisines' and 'dish\_liked'.

These features were combined into a single sparse matrix, and cosine similarity was calculated to quantify the similarity between restaurants.

## Usage

The generated cosine similarity matrix can be used to recommend restaurants that are similar to a restaurant a user likes or has previously visited. Further development would involve building a recommendation function that takes a restaurant as input and returns a list of similar restaurants based on the calculated similarity scores.

## Future Enhancements

*   Implement a recommendation function based on the cosine similarity matrix.
*   Explore other similarity metrics or recommendation algorithms (e.g., collaborative filtering).
*   Incorporate natural language processing (NLP) techniques for more advanced analysis of reviews and dish descriptions.
*   Build a user interface to interact with the recommendation system.
*   Evaluate the performance of the recommendation system using appropriate metrics.

## Description
Build a multi-input LSTM model that takes padded text sequences and other features to predict restaurant ratings and recommend similar restaurants.

## Features
* Building and training a multi-input LSTM model for restaurant rating prediction.
* Evaluating the trained model using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
* Finding similar restaurants based on the model's learned feature representations using Cosine Similarity.
* Saving and loading the trained model, tokenizer, and parameters for future use.

## Setup
To set up the project environment and run the notebook, follow these steps:

1.  **Mount Google Drive:** The notebook assumes the dataset and saved model components are stored in your Google Drive. You will need to mount your Drive using the following code:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
2.  **Install Dependencies:** Ensure you have the necessary libraries installed. The main libraries used are:
    *   `tensorflow`
    *   `keras`
    *   `numpy`
    *   `pandas`
    *   `scikit-learn`
    *   `pickle`
    *   `matplotlib` (for potential visualizations)
    *   `seaborn` (for potential visualizations)

    You can install them using pip:
    ```bash
    pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn
    ```
3.  **Load Data:** The notebook loads prepared data (padded sequences, other features, target, encoded dataframe, and tokenizer) from a specified directory in Google Drive (`/content/drive/MyDrive/restaurant_recommendation_data/`). Make sure these files are present in that location or update the `data_dir` variable accordingly.

## Usage
The notebook demonstrates the full workflow from data loading to model training and evaluation. To use the trained model for making predictions or finding similar restaurants in another notebook or script:

1.  **Mount Google Drive:** As in the setup, mount your Google Drive.
2.  **Load Model and Components:** Load the saved model, tokenizer, and parameters using the provided loading function (if implemented, or manually load using `tf.keras.models.load_model` and `pickle.load`).
3.  **Load Data:** Load the necessary data (`df_encoded`, `padded_sequences`, `other_features_input`) that was used to train the model, ensuring it's processed and aligned in the same way.
4.  **Use `find_similar_restaurants` function:** Utilize the `find_similar_restaurants` function, providing the restaurant name, loaded data, and the loaded model.

```python
# Example of loading and using the find_similar_restaurants function
# from google.colab import drive
# import os
# import pickle
# import tensorflow as tf
# from sklearn.metrics.pairwise import cosine_similarity
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# import pandas as pd

# drive.mount('/content/drive')

# # Define the directory where the model and components are saved
# save_dir = '/content/drive/MyDrive/restaurant_recommendation_model/'
# data_dir = '/content/drive/MyDrive/restaurant_recommendation_data/' # Define data_dir for loading data


# # Load the trained model
# loaded_model = tf.keras.models.load_model(os.path.join(save_dir, 'lstm_recommendation_model.keras'))

# # Load the tokenizer
# tokenizer_save_path = os.path.join(save_dir, 'tokenizer.pkl')
# with open(tokenizer_save_path, 'rb') as handle:
#     loaded_tokenizer = pickle.load(handle)

# # Load model parameters (including max_sequence_length if needed for padding)
# params_save_path = os.path.join(save_dir, 'model_params.pkl')
# with open(params_save_path, 'rb') as handle:
#     loaded_params = pickle.load(handle)
# max_sequence_length = loaded_params['max_sequence_length'] # Assuming max_sequence_length is saved

# # Load the data used for training (or processed data)
# # You would load the df_encoded_aligned.pkl, padded_sequences_aligned.npy, other_features_input_aligned.npy here
# # For this example, let's assume you have functions to load/reprocess data
# # Replace with your actual data loading/processing logic
# try:
#     # Example of loading data (adapt based on how you saved your data)
#     loaded_padded_sequences = np.load(os.path.join(data_dir, 'padded_sequences_aligned.npy'))
#     loaded_other_features_input = np.load(os.path.join(data_dir, 'other_features_input_aligned.npy'))
#     loaded_df_encoded = pd.read_pickle(os.path.join(data_dir, 'df_encoded_aligned.pkl'))
#     # Assuming df_features_aligned is needed for restaurant names, you might need to load/recreate it
#     loaded_df_features_aligned = loaded_df_encoded.copy().reset_index(drop=True)

# except FileNotFoundError as e:
#     print(f"Error loading necessary data files: {{e}}")
#     # Handle the error appropriately, perhaps exit or skip this part

# # Re-define the find_similar_restaurants function if it's not already available
# # (Copy the function definition from the notebook if needed)
# def find_similar_restaurants(restaurant_name, df_encoded, padded_sequences, model, n_top=10):
#     # ... (function implementation from the notebook)
#     # Ensure that inside this function, you use the passed df_encoded and padded_sequences,
#     # and recreate other_features_input from df_encoded as done during training.

#     # Ensure df_encoded and padded_sequences are aligned (assuming they are passed in aligned)
#     df_features_aligned = df_encoded.copy().reset_index(drop=True)

#     # Get the combined feature layer output from the model
#     # Use the correct layer name 'concatenate_4' based on the model summary
#     feature_layer_model = Model(inputs=model.input, outputs=model.get_layer('concatenate_4').output)

#     # Prepare the input data for the feature layer model
#     columns_to_exclude_from_other_features = ['rate', 'processed_reviews_text', 'name', 'dish_liked', 'processed_reviews', 'tokenized_reviews']
#     columns_for_other_features = df_encoded.columns.drop(columns_to_exclude_from_other_features, errors='ignore')

#     df_other_features_numeric = df_encoded[columns_for_other_features].copy()
#     for col in df_other_features_numeric.columns:
#         df_other_features_numeric[col] = pd.to_numeric(df_other_features_numeric[col], errors='coerce')
#     df_other_features_numeric.fillna(0, inplace=True)
#     other_features_input_aligned = df_other_features_numeric.values.astype(np.float32)


#     # Get the feature representations for all restaurants
#     all_restaurant_features = feature_layer_model.predict({{'text_input': padded_sequences, 'other_features_input': other_features_input_aligned}})

#     # Find the index of the input restaurant
#     if restaurant_name not in df_features_aligned['name'].values:
#         print(f"Restaurant '{{restaurant_name}}' not found in the dataset.")
#         return pd.DataFrame()

#     restaurant_index = df_features_aligned[df_features_aligned['name'] == restaurant_name].index[0]

#     # Get the feature vector for the input restaurant
#     input_restaurant_features = all_restaurant_features[restaurant_index].reshape(1, -1)

#     # Calculate cosine similarity between the input restaurant and all other restaurants
#     similarity_scores = cosine_similarity(input_restaurant_features, all_restaurant_features)

#     # Get the similarity scores for the input restaurant
#     similarity_scores = list(enumerate(similarity_scores[0]))

#     # Sort restaurants based on similarity scores in descending order
#     sorted_similar_restaurants = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

#     # Get the top-N similar restaurants (excluding the input restaurant itself)
#     top_similar_restaurants = []
#     for i in sorted_similar_restaurants:
#         if i[0] != restaurant_index:
#             top_similar_restaurants.append(i)
#         if len(top_similar_restaurants) >= n_top:
#             break

#     # Get the names and similarity scores of the top similar restaurants
#     top_restaurant_indices = [i[0] for i in top_similar_restaurants]
#     top_restaurant_names = df_features_aligned.loc[top_restaurant_indices, 'name'].values
#     top_restaurant_scores = [i[1] for i in top_similar_restaurants]

#     # Create a dataframe of similar restaurants
#     similar_restaurants_df = pd.DataFrame({{'Restaurant Name': top_restaurant_names,
#         'Similarity Score': top_restaurant_scores
#     }})

#     return similar_restaurants_df


# # Example usage:
# if 'loaded_df_encoded' in locals() and 'loaded_padded_sequences' in locals() and 'loaded_model' in locals():
#     restaurant_name_to_find_similar = 'Jalsa' # Replace with a valid restaurant name
#     similar_restaurants_df = find_similar_restaurants(
#         restaurant_name_to_find_similar,
#         loaded_df_encoded, # Use the loaded dataframe
#         loaded_padded_sequences, # Use the loaded padded sequences
#         loaded_model,
#         n_top=5
#     )
#     print(f"Top 5 similar restaurants to '{{restaurant_name_to_find_similar}}':")
#     print(similar_restaurants_df)
# else:
#     print("Data or model not loaded. Cannot run find_similar_restaurants example.")

Data
The project utilizes the Zomato Bangalore Restaurants Dataset. The dataset is sourced from Not explicitly mentioned, but loaded from '/content/drive/MyDrive/restaurant_recommendation_data/'. It contains information about restaurants in Bangalore, including name, rating, votes, dish liked, cost, reviews, location, restaurant type, and cuisines.

Data Preprocessing and Feature Engineering
The following key steps were performed to prepare the data for the multi-input LSTM model:

Loading and cleaning the 'rate' column, converting to numeric, and removing rows with missing ratings.
Dropping unnecessary columns.
Preprocessing 'reviews_list' by cleaning text, removing stopwords, tokenizing, and creating 'processed_reviews_text'.
Converting processed text into sequences and padding.
One-hot encoding categorical features.
Aligning padded text sequences and other features for multi-input model.
Model Architecture
The recommendation system is built using a multi-input LSTM model in TensorFlow/Keras. The model is designed to process both textual review data and structured restaurant features separately before combining them for prediction.

Text Input Branch:
Input Layer (shape: max_sequence_length)
Embedding Layer (input_dim: vocab_size, output_dim: 128)
LSTM Layer (units: lstm_units)
Other Features Input Branch:
Input Layer (shape: number of other features)
Dense Layer (units: dense_units, activation: 'relu')
Combination:
Concatenate layer combining LSTM output and Dense layer output
Prediction Layers:
Dense Layer (units: dense_units, activation: 'relu')
Output Dense Layer (units: 1, activation: 'linear' for regression)
Training
The model was trained using the following configuration:

Optimizer: Adam (learning_rate=learning_rate)
Loss Function: Mean Squared Error (MSE)
Metrics: Mean Absolute Error (MAE)
Evaluation
The model's performance was evaluated on a separate test dataset. The evaluation results are as follows:

Test Loss (MSE): 0.2145
Test MAE: 0.3047
These metrics indicate the model's accuracy in predicting restaurant ratings.

Finding Similar Restaurants
To find restaurants similar to a given restaurant, the project leverages the trained LSTM model's ability to generate a combined feature representation for each restaurant. The process involves:

Create a feature layer model to output combined features.
Get feature representations for all restaurants.
Find the index and feature vector of the input restaurant.
Calculate cosine similarity between the input restaurant and all others.
Sort and return the top-N most similar restaurants (excluding the input). The similarity is calculated using Cosine Similarity between the feature vector of the target restaurant and all other restaurants.
Saving and Loading the Model
The trained model and associated components are saved to Google Drive for persistence and ease of use in other environments. The components saved are:

Trained TensorFlow model ('lstm_recommendation_model.keras')
Fitted Tokenizer ('tokenizer.pkl')
Model parameters (e.g., max_sequence_length, vocab_size) ('model_params.pkl')
Paths to data files ('data_paths.pkl') Mount Google Drive and load the saved files from the specified directory.
Future Improvements
Potential areas for future work and improvements include:

Hyperparameter Tuning (LSTM units, dense units, dropout, optimizer, learning rate, epochs, batch size)
Text Feature Engineering (GloVe, FastText, more complex processing)
Feature Engineering for Other Columns (e.g., 'menu_item')
Different Model Architectures (GRU, attention, traditional ML)
Developing Recommendation Logic based on predictions
Incorporating User and Item Embeddings for a full recommendation system
Using Cross-validation for more robust performance estimation
Dependencies
The main dependencies required to run this project are:

tensorflow
keras
numpy
pandas
scikit-learn
pickle
matplotlib
seaborn
