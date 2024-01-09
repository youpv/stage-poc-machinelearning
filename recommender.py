### Algorithm 1

# from surprise import KNNWithMeans
# from load_data import data

# # To use item-based cosine similarity
# sim_options = {
#     "name": "cosine",
#     "user_based": False,  # Compute  similarities between items
# }
# algo = KNNWithMeans(sim_options=sim_options)
# trainingSet = data.build_full_trainset()
# algo.fit(trainingSet)
# prediction = algo.predict('E', 2)
# print(prediction.est)



### Algorithm 2

# from surprise import KNNWithMeans
# from surprise import Dataset
# from surprise.model_selection import GridSearchCV

# data = Dataset.load_builtin("ml-100k")
# sim_options = {
#     "name": ["msd", "cosine"],
#     "min_support": [3, 4, 5],
#     "user_based": [False, True],
# }

# param_grid = {"sim_options": sim_options}

# gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
# gs.fit(data)

# print(gs.best_score["rmse"])
# print(gs.best_params["rmse"])


### Algorithm 3

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import GridSearchCV

data = Dataset.load_builtin("ml-100k")

param_grid = {
    "n_epochs": [5, 10],
    "lr_all": [0.002, 0.005],
    "reg_all": [0.4, 0.6]
}
gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)

gs.fit(data)

print(gs.best_score["rmse"])
print(gs.best_params["rmse"])



# import pandas as pd
# from surprise import Dataset
# from surprise import Reader
# from surprise import SVD
# from surprise.model_selection import GridSearchCV

# # This is the same data that was plotted for similarity earlier
# # with one new user "E" who has rated only movie 1
# ratings_dict = {
#     "item": [1, 2, 1, 2, 1, 2, 1, 2, 1],
#     "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
#     "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5, 3],
# }

# df = pd.DataFrame(ratings_dict)
# reader = Reader(rating_scale=(1, 5))

# # Loads Pandas dataframe
# data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

# param_grid = {
#     "n_epochs": [5, 10],
#     "lr_all": [0.002, 0.005],
#     "reg_all": [0.4, 0.6]
# }
# gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3, refit=True)

# gs.fit(data)

# # Predict the score that user E would give movie 2
# print(gs.predict('E', 2))