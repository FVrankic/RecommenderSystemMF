import sys
import os

import numpy as np
from util.dataset import MovieLensDataset

import time
import logging

class RecommenderCF:
    def __init__(self, dataset: np.ndarray, variation: str = 'item-item', sim_mat: dict = None, c: int = 2, k: int = 5) -> None:
        """
        Initialize the Collaborative Filtering Recommender System.

        :param dataset: The dataset containing user-item interactions.
        :param variation: The type of collaborative filtering ('item-item' or 'user-user').
        :param sim_mat: Precomputed similarity matrix (optional).
        :param c: Minimum number of common ratings required for similarity computation.
        :param k: Number of most similar objects to consider in predictions.
        """
        self.name = 'Collaborative filtering Recommender System'
        self.data = dataset[:, :3]
        self.traindata = None  # Known data for usage
        self.validata = None  # Unknown data for validation
        self.variation = variation
        self.ratings_average = None
        self.users = np.unique(dataset[:, 0])
        self.items = np.unique(dataset[:, 1])
        self.users_average = None
        self.items_average = None

        self.rmse = list()
        self.mae = list()

        self.sim_mat = sim_mat

        self.ratings_users = None
        self.ratings_items = None

        self.c = c  # Number of minimum common ratings
        self.k = k  # Number of most similar objects taken into prediction calculation
        return

    def train_validation_split(self, r: float = 0.9) -> None:
        """
        Splits the full dataset into train and validation sets based on ratio `r`.

        :param r: Ratio of the dataset to be used for training.
        """
        print(f"Splitting data with a ratio of {r}...")
        np.random.shuffle(self.data)
        index = int(len(self.data) * r)
        self.traindata = self.data[:index, :]
        self.validata = self.data[index:, :]

        self.users = np.unique(self.traindata[:, 0])
        self.items = np.unique(self.traindata[:, 1])
        self.ratings_users = {user_id: self.traindata[self.traindata[:, 0] == user_id, 1:3] for user_id in self.users}
        self.ratings_items = {item_id: self.traindata[self.traindata[:, 1] == item_id, 0:3] for item_id in self.items}

        print(f"Data split completed: train set size = {len(self.traindata)}, validation set size = {len(self.validata)}")
        return

    def get_averages(self) -> None:
        """
        Calculates average ratings for users and items.
        """
        print("Calculating average ratings for users and items...")
        self.ratings_average = np.mean(self.traindata[:, 2])
        self.items_average = {item_id: np.mean(self.traindata[self.traindata[:, 1] == item_id, 2]) for item_id in self.items}
        self.users_average = {user_id: np.mean(self.traindata[self.traindata[:, 0] == user_id, 2]) for user_id in self.users}
        print(f"Global average rating: {self.ratings_average:.4f}")
        return

    def calculate_similarity(self, object1: int, object2: int) -> float:
        """
        Calculate the similarity between two objects (users or items) based on ratings.

        :param object1: ID of the first object.
        :param object2: ID of the second object.
        :return: The similarity score.
        """
        if self.variation == 'item-item':
            subjects1 = self.ratings_items[object1][:, 0]
            subjects2 = self.ratings_items[object2][:, 0]
        else:
            subjects1 = self.ratings_users[object1][:, 0]
            subjects2 = self.ratings_users[object2][:, 0]

        common_subjects = np.intersect1d(subjects1, subjects2)
        if len(common_subjects) < self.c:
            return 0

        ratings1 = list()
        ratings2 = list()
        for subject in common_subjects:
            if self.variation == 'item-item':
                rating1 = self.ratings_items[object1][self.ratings_items[object1][:, 0] == subject, 1]
                rating2 = self.ratings_items[object2][self.ratings_items[object2][:, 0] == subject, 1]
            else:
                rating1 = self.ratings_users[object1][self.ratings_users[object1][:, 0] == subject, 1]
                rating2 = self.ratings_users[object2][self.ratings_users[object2][:, 0] == subject, 1]

            if rating1.size > 0 and rating2.size > 0:
                ratings1.append(rating1[0])
                ratings2.append(rating2[0])

        ratings1 = np.array(ratings1)
        ratings2 = np.array(ratings2)

        mean1 = np.mean(ratings1)
        mean2 = np.mean(ratings2)
        num = np.sum((ratings1 - mean1) * (ratings2 - mean2))
        denom = np.sqrt(np.sum((ratings1 - mean1) ** 2) * np.sum((ratings2 - mean2) ** 2))

        return num / denom if denom != 0 else 0

    def get_prediction(self, item: int, user: int) -> float:
        """
        Predict the rating for a given user-item pair.

        :param item: The item ID.
        :param user: The user ID.
        :return: The predicted rating.
        """
        if self.variation == 'item-item':
            primary = item
            secondary = user
            if primary not in self.ratings_items:
                return self.ratings_average if self.ratings_average else 3.0
            iterdict = {obj: rating for obj, rating in self.ratings_users[secondary]}
        else:
            primary = user
            secondary = item
            if primary not in self.ratings_users:
                return self.ratings_average if self.ratings_average else 3.0
            if secondary not in self.ratings_items:
                return self.ratings_average if self.ratings_average else 3.0
            iterdict = {row[0]: row[1] for row in self.ratings_users[primary]}

        similarities = list()
        for subject in iterdict.keys():
            similarity = self.sim_mat.get((primary, subject), self.sim_mat.get((subject, primary), 0))
            similarities.append((similarity, iterdict[subject]))

        similarities = sorted(similarities, key=lambda x: -x[0])[:self.k]

        #adding bias
        bx = self.users_average.get(user, self.ratings_average) - self.ratings_average
        bi = self.items_average.get(item, self.ratings_average) - self.ratings_average
        bx_i = self.ratings_average + bx + bi

        num = sum(similarity * rating for similarity, rating in similarities if similarity > 0)
        denom = sum(abs(similarity) for similarity, _ in similarities if similarity > 0)

        if self.variation == 'item-item':
            return np.clip(bx_i if denom == 0 else bx_i + (num / denom), 1, 5)
        else:
            return np.clip(bx_i if denom == 0 else 0 + (num / denom), 1, 5)
            # return self.ratings_average if denom == 0 else num/denom
        # return np.clip(bx_i if denom == 0 else bx_i + (num / denom), 1, 5)


    def precompute_similarities(self) -> None:
        """
        Precompute the similarity matrix for all objects (users or items).
        """
        print("Precomputing similarity matrix started...")
        start = time.time()

        objects = self.items if self.variation == 'item-item' else self.users
        self.sim_mat = {}
        for n, obj1 in enumerate(objects):
            for obj2 in objects[n + 1:]:
                similarity = self.calculate_similarity(obj1, obj2)
                self.sim_mat[(obj1, obj2)] = similarity
                self.sim_mat[(obj2, obj1)] = similarity

        finish = time.time()
        print(f"Similarity matrix precomputed in {finish - start:.2f} seconds.")

    def get_evaluation(self) -> tuple[float, float]:
        """
        Evaluate the performance of the recommender system using RMSE and MAE.

        :return: RMSE and MAE scores.
        """
        print("Evaluating the recommender system...")
        start = time.time()

        predictions = [self.get_prediction(item, user) for user, item, _ in self.validata]
        ratings = self.validata[:, 2]

        rmse = np.sqrt(np.mean((np.array(predictions) - ratings) ** 2))
        mae = np.mean(np.abs(np.array(predictions) - ratings))

        self.rmse.append(rmse)
        self.mae.append(mae)

        finish = time.time()
        print(f"Evaluation completed in {finish - start:.2f} seconds.")
        return rmse, mae


if __name__ == "__main__":
    
    ds = MovieLensDataset()
    ds.load100k()

    recsys = RecommenderCF(ds.data, variation='item-item')
    recsys.train_validation_split(r=0.8)
    recsys.get_averages()
    recsys.precompute_similarities()

    rmse, mae = recsys.get_evaluation()
    print(f"Test Results - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
