import sys
import os

import numpy as np
from util.dataset import MovieLensDataset

import logging
import time


class RecommenderMF:

    def __init__(self, dataset: np.ndarray, variation: str, k: int = 10, i: int = 10, l: float = 0.004, e: float = 0.014):
        """
        Initializes the RecommenderMF class with given parameters.

        Args:
            dataset (np.ndarray): The dataset containing user-item-rating triplets.
            variation (str): The variation of the recommender system (e.g., 'sgd', 'als').
            k (int, optional): Number of latent factors. Defaults to 10.
            i (int, optional): Number of iterations. Defaults to 10.
            l (float, optional): Regularization factor (lambda). Defaults to 0.01.
            e (float, optional): Learning rate (eta). Defaults to 0.05.
        """
        self.name = 'Collaborative filtering Recommender System using Matrix Factorization'
        self.data = dataset[:, :3]
        self.variation = variation
        self.traindata = None
        self.validata = None

        self.user_factors = None
        self.item_factors = None
        # self.predictions = None
        self.rmse = list()
        self.mae = list()

        # Hyperparameters
        self.k = k  # number of latent factors
        self.i = i  # number of iterations
        self.l = l  # l as lambda, regularization factor
        self.e = e  # e as etha, learning rate

        # Latent factors
        nousers = int(np.max(self.data[:, 0])) + 1  # Maximum user ID (0-based ratings dataset)
        noitems = int(np.max(self.data[:, 1])) + 1  # Maximum item ID
        self.P = np.random.rand(nousers, self.k) * 0.01  # User factors
        self.Q = np.random.rand(noitems, self.k) * 0.01  # Item factors

        self.ratings_average = None
        self.b_u = np.zeros(nousers)
        self.b_i = np.zeros(noitems)

        return
    
    
    def train_validation_split(self, r: float = 0.9) -> None:
        """
        Splits the dataset into training and validation sets.

        Args:
            r (float, optional): Proportion of the data to use for training. Defaults to 0.9.
        """
        print(f"Splitting data with a ratio of {r}...")

        np.random.shuffle(self.data)  # Random
        index = int(len(self.data) * r)
        self.traindata = self.data[:index, :]  # Train data
        self.validata = self.data[index:, :]  # Validation data

        print(f"Data split completed: train set size = {len(self.traindata)}, validation set size = {len(self.validata)}")

        return
    
    
    def get_averages(self) -> None:
        """
        Calculates average ratings for users and items.
        """
        print("Calculating average ratings for users and items...")

        self.ratings_average = np.mean(self.traindata[:, 2])
        
        users = np.unique(self.traindata[:, 0]).astype(int)
        items = np.unique(self.traindata[:, 1]).astype(int)
        max_user_id = int(np.max(users)) + 1  # max user ID + 1
        max_item_id = int(np.max(items)) + 1  # max item ID + 1
        b_u = np.zeros(max_user_id, dtype=float)
        b_i = np.zeros(max_item_id, dtype=float)

        for user in users:
            users_ratings = self.traindata[self.traindata[:, 0] == user, 2]
            b_u[user] = np.mean(users_ratings) - self.ratings_average if users_ratings.size > 0 else 0

        for item in items:
            items_ratings = self.traindata[self.traindata[:, 1] == item, 2]
            b_i[item] = np.mean(items_ratings) - self.ratings_average if items_ratings.size > 0 else 0

        self.b_u = dict(zip(users, b_u[users]))
        self.b_i = dict(zip(items, b_i[items]))

        print(f"Global average rating: {self.ratings_average:.4f}")


    
    def sgd(self) -> None:
        """
        Performs stochastic gradient descent to optimize latent factors.
        """
        print("Stochastic gradient descent training started...")
        start = time.time()

        prev_loss = float('inf')

        for epoch in range(self.i):
            
            loss = 0
            
            for user, item, rating in self.traindata:
                prediction = self.get_prediction(user=int(user), item=int(item))
                error = rating - prediction
                
                self.P[int(user)] += self.e * (error * self.Q[int(item)] - self.l * self.P[int(user)])
                self.Q[int(item)] += self.e * (error * self.P[int(user)] - self.l * self.Q[int(item)])
                
                loss += error ** 2
            
            loss = loss / len(self.traindata)

            if abs(prev_loss - loss) < 0.00000001:  # 0.0001
                print(f"Converged after {epoch+1} epochs.")
                break

            prev_loss = loss

        finish = time.time()
        print(f"Stochastic gradient descent training finished in {finish - start:.2f} seconds.")

        return
    

    def als(self) -> None:
        """
        Perform Alternating Least Squares (ALS) to optimize latent factors.
        """

        print("Alternating Least Squares training started...")
        start = time.time()

        users, items = self.P.shape[0], self.Q.shape[0]
        R = np.zeros((users, items))
        for user, item, rating in self.traindata:
            R[int(user), int(item)] = rating

        for epoch in range(self.i):
            for u in range(users):
                R_u = R[u, :]
                non_zero_indices = R_u > 0
                Q_tilde = self.Q[non_zero_indices]
                R_tilde = R_u[non_zero_indices] - self.ratings_average - self.b_i.get(u, 0)

                self.P[u] = np.linalg.solve(Q_tilde.T @ Q_tilde + self.l * np.eye(self.k), Q_tilde.T @ R_tilde)

                # update bias
                self.b_u[u] = (np.sum(R_tilde - self.P[u] @ Q_tilde.T) / (len(R_tilde) + self.l)) if len(R_tilde) > 0 else 0

            for j in range(items):
                R_j = R[:, j]
                non_zero_indices = R_j > 0
                P_tilde = self.P[non_zero_indices]
                R_tilde = R_j[non_zero_indices] - self.ratings_average - self.b_u.get(j, 0)

                self.Q[j] = np.linalg.solve(P_tilde.T @ P_tilde + self.l * np.eye(self.k), P_tilde.T @ R_tilde)

                self.b_i[j] = (np.sum(R_tilde - P_tilde @ self.Q[j]) / (len(R_tilde) + self.l)) if len(R_tilde) > 0 else 0

            print(f"Epoch {epoch+1}/{self.i} completed.")

        finish = time.time()
        print(f"ALS training finished in {finish - start:.2f} seconds.")
        return

    

    def get_prediction(self, user: int, item: int) -> float:
        """
        Predicts the rating for a given user-item pair.

        Args:
            user (int): User ID.
            item (int): Item ID.

        Returns:
            float: Predicted rating.
        """
        # return self.ratings_average + self.b_u.get(user, 0) + self.b_i.get(item, 0) + np.dot(self.P[int(user)], self.Q[int(item)])
        return np.clip(self.ratings_average + self.b_u.get(user, 0) + self.b_i.get(item, 0) + np.dot(self.P[int(user)], self.Q[int(item)]), 1, 5)
    
    
    def get_evaluation(self) -> tuple[float, float]:
        """
        Evaluates the recommender system on the validation set.

        Returns:
            tuple[float, float]: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
        """
        print("Evaluating the recommender system...")
        start = time.time()

        err = list()
        abserr = list()

        for user, item, rating in self.validata:
            prediction = self.get_prediction(int(user), int(item))
            err.append((rating - prediction)**2)
            abserr.append(abs(rating - prediction))

        rmse = np.sqrt(np.mean(err))
        mae = np.mean(abserr)
        self.rmse.append(rmse)
        self.mae.append(mae)

        finish = time.time()
        print(f"Evaluation completed in {finish - start:.2f} seconds.")

        return rmse, mae
    

if __name__ == "__main__":

    ds = MovieLensDataset()

    ds.load100k()
    recsys = RecommenderMF(dataset=ds.data, variation='sgd', l=0.004, e=0.014, i=10)

    ds.load1m()
    recsys = RecommenderMF(dataset=ds.data, variation='sgd', l=0.004, e=0.014, i=10)

    recsys.train_validation_split(r=0.8)
    recsys.get_averages()
    recsys.sgd()

    rmse, mae = recsys.get_evaluation()
    print(f"Test Results - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    print(f"Test Results - RMSE: {np.mean(recsys.rmse):.4f}, MAE: {np.mean(recsys.mae):.4f}")
    
    recsys = RecommenderMF(dataset=ds.data, variation='als', l=0.1, e=0.001, i=10)
    recsys.train_validation_split(r=0.8)
    recsys.get_averages()
    recsys.als()

    rmse, mae = recsys.get_evaluation()
    print(f"Test Results - RMSE: {np.mean(recsys.rmse):.4f}, MAE: {np.mean(recsys.mae):.4f}")

    # itemitem: 1.122, 0.9412
    # useruser: 1.157, 0.9202
    # sgd: 1.0485, 0.83
    # als: ,
