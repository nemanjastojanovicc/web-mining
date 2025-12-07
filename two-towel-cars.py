# import pandas as pd
# import numpy as np

# https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
# df = pd.read_csv("Car details v3.csv")
# df.head()

# # clear data
# df = df[['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission']]

# df.dropna(inplace=True)

# # Pretvaranje kategorija u numeric
# df['fuel'] = df['fuel'].astype('category').cat.codes
# df['seller_type'] = df['seller_type'].astype('category').cat.codes
# df['transmission'] = df['transmission'].astype('category').cat.codes

# # normalizacija numerickih atributa
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# df[['year','selling_price','km_driven']] = scaler.fit_transform(df[['year','selling_price','km_driven']])

# # kreiranje item feature matrice
# item_features = df[['year','selling_price','km_driven','fuel','seller_type','transmission']].values
# item_features.shape

# # simulacija korisnickog profila
# user_profile = {
#     "year": 0.8,               # preferira novija kola
#     "selling_price": 0.4,      # srednji budžet
#     "km_driven": 0.2,          # mala kilometraža
#     "fuel": 0,                 # 0 = Benzin (u kodiranju)
#     "seller_type": 0,          # svejedno
#     "transmission": 1          # 1 = Automatika
# }

# user_vec = np.array(list(user_profile.values())).reshape(1, -1)

# # Two-Tower Model (minimalni TensorFlow MVP)

# # user network
# import tensorflow as tf
# from tensorflow.keras import layers, Model

# embedding_dim = 16

# # USER tower
# user_input = layers.Input(shape=(6,))
# u = layers.Dense(32, activation='relu')(user_input)
# u = layers.Dense(embedding_dim)(u)
# user_tower = Model(user_input, u)

# # item network
# item_input = layers.Input(shape=(6,))
# i = layers.Dense(32, activation='relu')(item_input)
# i = layers.Dense(embedding_dim)(i)
# item_tower = Model(item_input, i)

# # Loss funkcija (dot product)
# user_emb = user_tower(user_input)
# item_emb = item_tower(item_input)

# dot = layers.Dot(axes=1)([user_emb, item_emb])

# model = Model(inputs=[user_input, item_input], outputs=dot)
# model.compile(optimizer='adam', loss='mse')

# ## Formiranje pozitivnih i negativnih parova

# # pozitivni parovi = automobili čije karakteristike liče na korisničke preference
# # negativni parovi = potpuno suprotne karakteristike

# # Pozitivni uzorci = automobili sa automatikom i benzinom
# positive_items = df[(df['fuel'] == user_profile['fuel']) &
#                     (df['transmission'] == user_profile['transmission'])]

# # Negativni uzorci
# negative_items = df[(df['fuel'] != user_profile['fuel'])]

# pos_samples = positive_items.sample(200, replace=True)[['year','selling_price','km_driven','fuel','seller_type','transmission']].values
# neg_samples = negative_items.sample(200, replace=True)[['year','selling_price','km_driven','fuel','seller_type','transmission']].values

# X_user = np.vstack([
#     np.repeat(user_vec, len(pos_samples), axis=0),
#     np.repeat(user_vec, len(neg_samples), axis=0)
# ])

# X_item = np.vstack([pos_samples, neg_samples])

# y = np.hstack([np.ones(len(pos_samples)), np.zeros(len(neg_samples))])

# ## Trening modela
# model.fit([X_user, X_item], y, epochs=5, batch_size=32)

# ## Generisanje embeddinga za sve automobile
# item_emb_matrix = item_tower.predict(item_features)
# user_embedding = user_tower.predict(user_vec)

# ## Top N poruka
# from sklearn.metrics.pairwise import cosine_similarity

# scores = cosine_similarity(user_embedding, item_emb_matrix)[0]
# top_n_idx = np.argsort(scores)[::-1][:10]

# recommended_cars = df.iloc[top_n_idx]
# recommended_cars[['name','year','selling_price','km_driven']]

