# WARNING: cca 6 minutes runtime

from surprise import Dataset, KNNWithMeans, SVD, accuracy
from surprise.model_selection import train_test_split
from prettytable import PrettyTable
import time


start = time.time()
data_100k = Dataset.load_builtin('ml-100k')
data_100k_name = "100k"
finish = time.time()
print(f"Loaded MovieLens 100k dataset in {finish - start:.2f} seconds.")

start = time.time()
data_1m = Dataset.load_builtin('ml-1m')
data_1m_name = "1m"
finish = time.time()
print(f"Loaded MovieLens 1M dataset in {finish - start:.2f} seconds.")

r = 0.2
item_item_options = {
    'name': 'pearson',
    'user_based': False,
    'min_support': 2
}
user_user_options = {
    'name': 'pearson',
    'user_based': True,
    'min_support': 2
}

k = 5
iterations = 1

datasets = [
    (data_100k, data_100k_name),
    (data_1m, data_1m_name)
]
options = [
    (item_item_options, "item-item", "KNNWithMeans"),
    (user_user_options, "user-user", "KNNWithMeans"),
    (None, None , "SVD")
]

results = list()

for dataset in datasets:
    for option in options:
        total_time = 0
        total_rmse = 0
        total_mae = 0

        data, data_name = dataset
        sim_option, variation, algo = option
        print(f"\nTrain and evaluate on {algo} {variation} on MovieLens {data_name}")

        for i in range(iterations):
            print(f"Iteration {i+1}/{iterations}")

            if algo == "KNNWithMeans":
                recommenderCF = KNNWithMeans(k=k, sim_options=sim_option)
            elif algo == "SVD":
                recommenderCF = SVD()

            traindata, validata = train_test_split(data, test_size=r)

            start = time.time()
            recommenderCF.fit(traindata)
            
            predictions = recommenderCF.test(validata)

            rmse = accuracy.rmse(predictions, verbose=False)
            mae = accuracy.mae(predictions, verbose=False)
            total_rmse += rmse
            total_mae += mae

            end = time.time()
            runtime = end - start
            total_time += runtime
            print(f"Training and evaluation completed in {runtime:.2f} seconds.")
            print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        avg_time = total_time / iterations
        avg_rmse = total_rmse / iterations
        avg_mae = total_mae / iterations

        recommender_type = "RecommenderMF" if algo == "SVD" else "RecommenderCF"
        print(f"\n{recommender_type} {variation} on MovieLens {data_name}")
        print(f"Average Runtime: {avg_time:.2f} seconds")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Average MAE: {avg_mae:.4f}\n")

        results.append({
            "Recommender Type": recommender_type,
            "Variation": variation,
            "Dataset": data_name,
            "Avg RMSE": avg_rmse,
            "Avg MAE": avg_mae,
            "Avg Time": avg_time
        })


table = PrettyTable()
table.field_names = ["Recommender Type", "Variation", "Dataset", "Avg RMSE", "Avg MAE", "Avg Time (s)"]

for result in results:
    table.add_row([
        result["Recommender Type"],
        result["Variation"],
        result["Dataset"],
        f"{result['Avg RMSE']:.4f}",
        f"{result['Avg MAE']:.4f}",
        f"{result['Avg Time']:.2f}"
    ])

print("\nSummary of Results:")
print(table)
