# WARNING: cca 100 minute runtime

import sys
import os
import time

from util.dataset import MovieLensDataset
from models.recommenderCF import RecommenderCF
from models.recommenderMF import RecommenderMF
from evaluation.evaluator import evaluate_model

from prettytable import PrettyTable


def print_results_table(results):

    table = PrettyTable()
    table.field_names = ["Recommender Type", "Variation", "Dataset", "Avg RMSE", "Avg MAE", "Avg Time"]

    for key, stats in results["combinations"].items():
        parts = key.split("-")

        if len(parts) < 3:
            print(f"Error: Unexpected key format: {key}")
            continue

        recommender_type = parts[0]
        dataset = parts[-1]
        variation = "-".join(parts[1:-1])

        avg_rmse = stats["total_rmse"] / stats["count"]
        avg_mae = stats["total_mae"] / stats["count"]
        avg_time = stats["total_time"] / stats["count"]

        table.add_row([recommender_type, variation, dataset, f"{avg_rmse:.4f}", f"{avg_mae:.4f}", f"{avg_time:.2f} s"])

    print(table)


def save_results_to_file(results, data="results/results_history.txt"):
    """
    Save results to a file, updating averages and tracking the best RMSE and MAE.
    """
    best_rmse = results.get("best_rmse")
    best_mae = results.get("best_mae")
    combinations = results.get("combinations")

    os.makedirs("results", exist_ok=True) 

    if os.path.exists(data):
        with open(data, "r") as file:
            lines = file.readlines()

        existing = {}
        for line in lines:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                existing[key.strip()] = value.strip()

        for key, values in combinations.items():
            count_key = f"{key}-Count"
            avg_rmse_key = f"{key}-Average RMSE"
            avg_mae_key = f"{key}-Average MAE"
            avg_time_key = f"{key}-Average Time"

            previous_count = int(existing.get(count_key, 0))
            previous_avg_rmse = float(existing.get(avg_rmse_key, 0.0))
            previous_avg_mae = float(existing.get(avg_mae_key, 0.0))
            previous_avg_time = float(existing.get(avg_time_key, 0.0))

            new_count = previous_count + values["count"]
            updated_avg_rmse = (previous_avg_rmse * previous_count + values["total_rmse"]) / new_count
            updated_avg_mae = (previous_avg_mae * previous_count + values["total_mae"]) / new_count
            updated_avg_time = (previous_avg_time * previous_count + values["total_time"]) / new_count

            existing[count_key] = str(new_count)
            existing[avg_rmse_key] = f"{updated_avg_rmse:.4f}"
            existing[avg_mae_key] = f"{updated_avg_mae:.4f}"
            existing[avg_time_key] = f"{updated_avg_time:.4f}"

        current_best_rmse = float(existing.get("Best RMSE Value", float("inf")))
        if best_rmse["value"] < current_best_rmse:
            existing["Best RMSE"] = f"{best_rmse['value']} (Recommender: {best_rmse['name']}, Variation: {best_rmse['variation']}, Dataset: {best_rmse['dataset']})"
            existing["Best RMSE Value"] = f"{best_rmse['value']}"

        current_best_mae = float(existing.get("Best MAE Value", float("inf")))
        if best_mae["value"] < current_best_mae:
            existing["Best MAE"] = f"{best_mae['value']} (Recommender: {best_mae['name']}, Variation: {best_mae['variation']}, Dataset: {best_mae['dataset']})"
            existing["Best MAE Value"] = f"{best_mae['value']}"

    else:
        existing = {}
        for key, values in combinations.items():
            count_key = f"{key}-Count"
            avg_rmse_key = f"{key}-Average RMSE"
            avg_mae_key = f"{key}-Average MAE"
            avg_time_key = f"{key}-Average Time"

            existing[count_key] = str(values["count"])
            existing[avg_rmse_key] = f"{values['total_rmse']:.4f}"
            existing[avg_mae_key] = f"{values['total_mae']:.4f}"
            existing[avg_time_key] = f"{values['total_time']:.4f}"

        existing["Best RMSE"] = f"{best_rmse['value']} (Recommender: {best_rmse['name']}, Variation: {best_rmse['variation']}, Dataset: {best_rmse['dataset']})"
        existing["Best RMSE Value"] = f"{best_rmse['value']}"
        existing["Best MAE"] = f"{best_mae['value']} (Recommender: {best_mae['name']}, Variation: {best_mae['variation']}, Dataset: {best_mae['dataset']})"
        existing["Best MAE Value"] = f"{best_mae['value']}"

    with open(data, "w") as file:
        for key, value in existing.items():
            file.write(f"{key}: {value}\n")
    print(f"Results updated in {data}")


def main():
    print("Loading MovieLens datasets...")
    ds = MovieLensDataset()

    start = time.time()
    ds.load100k()
    finish = time.time()
    print(f"Loaded MovieLens 100k dataset in {finish - start:.2f} seconds.")
    data_100k = ds.data
    data_100k_name = "100k"

    start = time.time()
    ds.load1m()
    finish = time.time()
    print(f"Loaded MovieLens 1m dataset in {finish - start:.2f} seconds.")
    data_1m = ds.data
    data_1m_name = "1m"

    recommenders = [
        {"name": "RecommenderCF", "class": RecommenderCF, "variation": "item-item", "data": data_100k, "dataset_name": data_100k_name},
        # {"name": "RecommenderCF", "class": RecommenderCF, "variation": "item-item", "data": data_1m, "dataset_name": data_1m_name},
        {"name": "RecommenderCF", "class": RecommenderCF, "variation": "user-user", "data": data_100k, "dataset_name": data_100k_name},
        # {"name": "RecommenderCF", "class": RecommenderCF, "variation": "user-user", "data": data_1m, "dataset_name": data_1m_name},
        # {"name": "RecommenderMF", "class": RecommenderMF, "variation": "sgd", "data": data_100k, "dataset_name": data_100k_name},
        # {"name": "RecommenderMF", "class": RecommenderMF, "variation": "sgd", "data": data_1m, "dataset_name": data_1m_name},
        # {"name": "RecommenderMF", "class": RecommenderMF, "variation": "als", "data": data_100k, "dataset_name": data_100k_name},
        # {"name": "RecommenderMF", "class": RecommenderMF, "variation": "als", "data": data_1m, "dataset_name": data_1m_name},
    ]

    results = {"combinations": {}, "best_rmse": {"value": float("inf")}, "best_mae": {"value": float("inf")}}

    for rec in recommenders:
        recommender_type = rec["class"].__name__
        variation = rec["variation"]
        dataset_name = rec["dataset_name"]

        key = f"{recommender_type}-{variation}-{dataset_name}"

        if key not in results["combinations"]:
            results["combinations"][key] = {"total_rmse": 0, "total_mae": 0, "total_time": 0, "count": 0}

        start = time.time()
        rmse, mae = evaluate_model(
            recommender=rec["class"],
            data=rec["data"],
            variation=rec["variation"],
            r=0.8,
            iterations=1
        )
        finish = time.time()

        time_spent = finish - start
        results["combinations"][key]["total_rmse"] += rmse
        results["combinations"][key]["total_mae"] += mae
        results["combinations"][key]["total_time"] += time_spent
        results["combinations"][key]["count"] += 1

        if rmse < results["best_rmse"]["value"]:
            results["best_rmse"] = {"value": rmse, "name": rec["name"], "variation": rec["variation"], "dataset": rec["dataset_name"]}
        if mae < results["best_mae"]["value"]:
            results["best_mae"] = {"value": mae, "name": rec["name"], "variation": rec["variation"], "dataset": rec["dataset_name"]}

    save_results_to_file(results)
    print_results_table(results)


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    main()
