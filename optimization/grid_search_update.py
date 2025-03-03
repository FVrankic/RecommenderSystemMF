from util.dataset import MovieLensDataset
from models.recommenderMF import RecommenderMF
import numpy as np

import time

def grid_search(dataset, variations, lambdas, ethas, k=10, i=10, r=0.8, iterations=5):
    """
    Performs grid search over lambda and eta hyperparameters for different variations of a recommender system.

    Parameters:
    - dataset: The dataset to use.
    - variations: List of recommender variations (e.g., ["sgd", "als"]).
    - lambdas: List of lambda values to test.
    - ethas: List of eta values to test.
    - k: Number of latent factors.
    - i: Number of iterations for MF.
    - r: Train-validation split ratio.
    - iterations: Number of times to train each parameter combination (to get stable averages).

    Returns:
    - best_results: Dictionary storing the best RMSE and MAE parameters per variation.
    """
    
    best_results = {variation: {"rmse": float('inf'), "mae": float('inf'), "params": None} for variation in variations}

    for variation in variations:
        print(f"\nStarting grid search for variation: {variation}")

        for l in lambdas:
            for e in ethas:
                rmse_scores = []
                mae_scores = []

                for _ in range(iterations):
                    recsys = RecommenderMF(dataset=dataset, variation=variation, k=k, i=i, l=l, e=e)
                    recsys.train_validation_split(r=r)
                    recsys.get_averages()

                    if variation == 'sgd':
                        recsys.sgd()
                    elif variation == 'als':
                        recsys.als()

                    rmse, mae = recsys.get_evaluation()
                    rmse_scores.append(rmse)
                    mae_scores.append(mae)

                avg_rmse = np.mean(rmse_scores)
                avg_mae = np.mean(mae_scores)

                print(f"Variation: {variation}, Lambda: {l}, Eta: {e}, Avg RMSE: {avg_rmse:.4f}, Avg MAE: {avg_mae:.4f}")

                if avg_rmse < best_results[variation]["rmse"]:
                    best_results[variation]["rmse"] = avg_rmse
                    best_results[variation]["params"] = {'l': l, 'e': e}

                if avg_mae < best_results[variation]["mae"]:
                    best_results[variation]["mae"] = avg_mae

        print(f"\nBest parameters for {variation}: RMSE = {best_results[variation]['rmse']:.4f}, "
              f"MAE = {best_results[variation]['mae']:.4f}, "
              f"Lambda: {best_results[variation]['params']['l']}, Eta: {best_results[variation]['params']['e']}")

    print("\n--- Final Best Parameters ---")
    for variation, results in best_results.items():
        print(f"Variation: {variation}")
        print(f"Best RMSE: {results['rmse']:.4f}, Best MAE: {results['mae']:.4f}")
        print(f"Best Parameters: Lambda (l) = {results['params']['l']}, Eta (e) = {results['params']['e']}")
        print("------")

    return best_results


if __name__ == "__main__":
    dataset = MovieLensDataset()
    dataset.load100k()

    variations = ["sgd", "als"] #SGD = 0.004, 0.014
    # lambdas = [0.005, 0.007, 0.01, 0.012, 0.015]
    lambdas = [0.003, 0.004, 0.005] #                        za als 0.003, 0.008; 0.008, 0.007; 0.007, 0.009; 0.005,0.009;  0.005, 0.01; 0.003, 0.011
    # ethas = [0.003, 0.004, 0.005, 0.006, 0.007]
    ethas = [0.013, 0.014, 0.015]
    start = time.time()
    results = grid_search(dataset.data, variations, lambdas, ethas, k=10, i=10, r=0.8, iterations=5)
    finish = time.time()
    print(f"Grid search done in {finish-start} seconds.")
