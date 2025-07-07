import pandas as pd
# load
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
tags = pd.read_csv("tags.csv")
# Merge
data = pd.merge(ratings, movies, on="movieId")
data = pd.merge(data, tags[['movieId', 'tag']], on="movieId", how="left")
# drop
data.drop(columns=["timestamp"], inplace=True)

data["tag"] = data["tag"].fillna("")
data = data.groupby(["userId", "movieId", "rating", "title", "genres"])["tag"].apply(lambda x: " ".join(x)).reset_index()

print("\nFinal Preprocessed Data:\n", data.head())

# prints the total no.of rows
# print(data.shape[0])

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
#NCF MODEL
# Encode user and movie IDs
data['userId'] = data['userId'].astype('category').cat.codes
data['movieId'] = data['movieId'].astype('category').cat.codes

# Prepare data for NCF
user_ids = data['userId'].values
movie_ids = data['movieId'].values
ratings = data['rating'].values

# Split data for NCF
X = np.array(list(zip(user_ids, movie_ids)))
y = ratings
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get unique users and movies
num_users = data['userId'].nunique()
num_movies = data['movieId'].nunique()

# Define input layers
user_input = layers.Input(shape=(1,), name='user_input')
movie_input = layers.Input(shape=(1,), name='movie_input')

# Embedding layers
user_embedding = layers.Embedding(input_dim=num_users + 1, output_dim=50, name='user_embedding')(user_input)
movie_embedding = layers.Embedding(input_dim=num_movies + 1, output_dim=50, name='movie_embedding')(movie_input)

# Flatten embeddings
user_vector = layers.Flatten()(user_embedding)
movie_vector = layers.Flatten()(movie_embedding)

# Concatenate user and movie vectors
concat = layers.Concatenate()([user_vector, movie_vector])

# Dense layers (MLP)
dense1 = layers.Dense(128, activation='relu')(concat)
dense2 = layers.Dense(64, activation='relu')(dense1)
dense3 = layers.Dense(32, activation='relu')(dense2)

# Output layer
ncf_output = layers.Dense(1)(dense3)

# Define model
ncf_model = keras.Model(inputs=[user_input, movie_input], outputs=ncf_output)

# Compile model
ncf_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

ncf_model.summary()

#trainig part
history = ncf_model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=10,batch_size=32,validation_split=0.2)

#evaluation of ncf
test_loss, test_mae = ncf_model.evaluate([X_test[:, 0], X_test[:, 1]], y_test)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

#AUTOENCODERS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense
# Create interaction matrix for Autoencoder
interaction_matrix = data.pivot(index='userId', columns='movieId', values='rating').fillna(0).values
num_users, num_movies = interaction_matrix.shape
# Normalize the data
interaction_matrix = interaction_matrix / np.max(interaction_matrix)

# Split data (80% train, 20% test)
train_size = int(num_users * 0.8)
train_data = interaction_matrix[:train_size]
test_data = interaction_matrix[train_size:]

train_data, test_data = train_test_split(interaction_matrix, test_size=0.2, random_state=42)


# Define Autoencoder
# Define input layer
input_layer = Input(shape=(num_movies,))

# Encoder
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# Bottleneck (latent space)
bottleneck = Dense(16, activation='relu')(encoded)

# Decoder
decoded = Dense(32, activation='relu')(bottleneck)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
output_layer = Dense(num_movies, activation='sigmoid')(decoded)

# Define model
autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)

# Compile model
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
autoencoder.summary()
#training part of autoencoder
history = autoencoder.fit(train_data,train_data,epochs=20,batch_size=32,validation_data=(test_data, test_data),verbose=1)
# evaluation of teh model
test_loss, test_mae = autoencoder.evaluate(test_data, test_data)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Save the trained model
autoencoder.save('autoencoder_model.h5', save_format='h5')
autoencoder = tf.keras.models.load_model('autoencoder_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# Get reconstructed ratings
reconstructed_ratings = autoencoder.predict(test_data)
#FINAL HYBRID MODEL

from sklearn.feature_extraction.text import TfidfVectorizer

# Combine genres and tags into one feature
data['content'] = data['genres'] + " " + data['tag']

# Create TF-IDF matrix for content features
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
content_features = tfidf.fit_transform(data['content']).toarray()

print("Content Features Shape:", content_features.shape)

# Get reconstructed ratings 
print("Reconstructed Ratings Shape:", reconstructed_ratings.shape)
# Get predictions from NCF model 
ncf_predictions = ncf_model.predict([X_test[:, 0], X_test[:, 1]])
print("NCF Predictions Shape:", ncf_predictions.shape)
#test data
#content fatures
content_train, content_test = content_features[:train_size], content_features[train_size:]
content_test = content_test[:ncf_predictions.shape[0]]
print("Content Test Shape:", content_test.shape)

#reconstructed feATURES
# Get unique user IDs for the test split of interaction_matrix
test_user_ids = np.arange(train_size, num_users)

# Create a dictionary to map actual user IDs to row indices in reconstructed_ratings
user_id_to_index = {user_id: idx for idx, user_id in enumerate(test_user_ids)}

reconstructed_test_flat = []

for user_id in X_test[:, 0]:
    # Get the correct row from reconstructed ratings
    if user_id in user_id_to_index:
        reconstructed_test_flat.append(reconstructed_ratings[user_id_to_index[user_id]])
    else:
        # If user_id is not found, append zeros to maintain shape
        reconstructed_test_flat.append(np.zeros(num_movies))
# Convert to numpy array
reconstructed_test_flat = np.array(reconstructed_test_flat)
print("Reconstructed Test Shape After Flattening:", reconstructed_test_flat.shape)

# Combine all features for the hybrid model
final_input = np.concatenate((ncf_predictions, reconstructed_test_flat, content_test), axis=1)
print("Final Hybrid Input Shape:", final_input.shape)

# Define Hybrid Model
input_layer = layers.Input(shape=(final_input.shape[1],))

# Dense layers for learning
dense1 = layers.Dense(128, activation='relu')(input_layer)
dense2 = layers.Dense(64, activation='relu')(dense1)
dense3 = layers.Dense(32, activation='relu')(dense2)

# Output layer for ratings
output_layer = layers.Dense(1)(dense3)

# Compile model
hybrid_model = keras.Model(inputs=input_layer, outputs=output_layer)
hybrid_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

hybrid_model.summary()

# Split train-test for hybrid
hybrid_train_size = int(final_input.shape[0] * 0.8)
hybrid_X_train = final_input[:hybrid_train_size]
hybrid_X_test = final_input[hybrid_train_size:]
hybrid_y_train = y_test[:hybrid_train_size]
hybrid_y_test = y_test[hybrid_train_size:]

# Train the hybrid model
history = hybrid_model.fit(hybrid_X_train, hybrid_y_train,epochs=10,batch_size=32,validation_split=0.2)


#printing the PREDICTED MOVIES BY THE  MODEL
# Get predictions from the hybrid model
predicted_ratings = hybrid_model.predict(hybrid_X_test)
# Ensure 'movieId' is categorical
# Ensure 'movieId' is categorical
data['movieId'] = data['movieId'].astype('category')

# Create a mapping for movieId to title
movie_id_to_title = dict(zip(data['movieId'].cat.codes, data['title']))

# Get predictions from the hybrid model
predicted_ratings = hybrid_model.predict(hybrid_X_test)

# Get top N recommendations (e.g., 5)
top_n = 5
recommendations = []

# For each user, get top N movie recommendations
for i in range(hybrid_X_test.shape[0]):
    user_id = X_test[hybrid_train_size + i, 0]  # Get user ID
    movie_id = X_test[hybrid_train_size + i, 1]  # Get corresponding movie ID
    
    # Get the predicted rating
    predicted_rating = predicted_ratings[i][0]  # Access the scalar value
    
    # Append to recommendations
    recommendations.append((user_id, movie_id, predicted_rating))

# Create a DataFrame for easy viewing
recommendations_df = pd.DataFrame(recommendations, columns=['userId', 'movieId', 'predicted_rating'])

# Sort by highest predicted rating per user
top_recommendations = recommendations_df.groupby('userId').apply(
    lambda x: x.sort_values('predicted_rating', ascending=False).head(top_n)
).reset_index(drop=True)

# Map movie IDs to titles
top_recommendations['title'] = top_recommendations['movieId'].map(movie_id_to_title)

# Display the top 20 recommendations
print(top_recommendations[['userId', 'title', 'predicted_rating']].head(20))

# Evaluate the hybrid model
test_loss, test_mae = hybrid_model.evaluate(hybrid_X_test, hybrid_y_test)
print(f"Hybrid Model Test Loss (MSE): {test_loss:.4f}")
print(f"Hybrid Model Test MAE: {test_mae:.4f}")
