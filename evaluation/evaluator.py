import sys
import os

import numpy as np
from util.dataset import MovieLensDataset
from models.recommenderCF import RecommenderCF
from models.recommenderMF import RecommenderMF

import time

def evaluate_model(recommender: type, variation: str, data: np.ndarray, iterations: int = 5, r: float = 0.9, **kwargs) -> tuple[float, float]:
    """
    Evaluate a recommender model over multiple iterations.

    Args:
        recommender (type): The recommender class to evaluate (RecommenderMF or RecommenderCF).
        variation (str): The variation of the recommender model to use.
        data (np.ndarray): The dataset to use for evaluation.
        iterations (int, optional): Number of iterations for evaluation. Defaults to 5.
        r (float, optional): The ratio for train-validation split. Defaults to 0.9.
        **kwargs: Additional parameters for the recommender constructor.

    Returns:
        tuple[float, float]: A tuple containing the average RMSE and MAE over iterations.
    """
    rmse_all = list()
    mae_all = list()

    start = time.time()

    for _ in range(iterations):

        if recommender.__name__ == "RecommenderCF":
            recsys = recommender(dataset=data, variation=variation, **kwargs)
        elif recommender.__name__ == "RecommenderMF":
            recsys = recommender(dataset=data, variation=variation, **kwargs)
        else:
            raise ValueError("Unsupported recommender type.")

        recsys.train_validation_split(r=r)
        recsys.get_averages()

        if hasattr(recsys, "precompute_similarities"):
            recsys.precompute_similarities()
        elif hasattr(recsys, "sgd") and recsys.variation == 'sgd':
            recsys.sgd()
        elif hasattr(recsys, "als") and recsys.variation == 'als':
            recsys.als()
        else:
            continue

        rmse, mae = recsys.get_evaluation()
        rmse_all.append(rmse)
        mae_all.append(mae)

    finish = time.time()
    print(f"\nAverage results for {recsys.name}, variation {recsys.variation}: "
          f"RMSE = {np.mean(rmse_all):.4f}, MAE = {np.mean(mae_all):.4f}")
    print(f"Evaluation finished in {finish - start:.2f} seconds.\n")

    return np.mean(rmse_all), np.mean(mae_all)


if __name__ == "__main__":
    
    ds = MovieLensDataset()
    ds.load100k()
    ds100k = ds.data
    ds.load1m()
    ds1m = ds.data

    # Evaluate item-item recommender
    print("Evaluating item-item recommender...")
    evaluate_model(
        recommender=RecommenderCF, 
        variation='item-item', 
        data=ds100k, 
        iterations=5, 
        r=0.8
    )

    # Evaluate user-user recommender
    print("Evaluating user-user recommender...")
    evaluate_model(
        recommender=RecommenderCF, 
        variation='user-user', 
        data=ds100k, 
        iterations=5, 
        r=0.8
    )

    # Evaluate mf-sgd recommender
    print("Evaluating mf-sgd recommender...")
    evaluate_model(
        recommender=RecommenderMF, 
        variation='sgd', 
        iterations=1, 
        r=0.8, 
        data=ds100k, 
        l=0.007, 
        e=0.003, 
        i=300
    )

    # Evaluate mf-als recommender
    print("Evaluating mf-als recommender...")
    evaluate_model(
        recommender=RecommenderMF, 
        variation='als', 
        iterations=1, 
        r=0.8, 
        data=ds1m, 
        l=0.015, 
        e=0.003, 
        i=300
    )
