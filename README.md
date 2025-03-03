# Movie Recommendation System

This project is a movie recommendation system using collaborative filtering and matrix factorization techniques. It processes datasets, evaluates models, and provides recommendations based on user preferences.

## 📂 Project Structure
```
C:.
|   main.py
|   requirements.txt
|   __init__.py
|
+---datasets
|   +---ml-100k
|   |       (MovieLens 100k dataset files)
|   |
|   \---ml-1m
|           (MovieLens 1M dataset files)
|
+---evaluation
|       evaluator.py
|       surprise_evaluation.py
|
+---loggings
|       dataset.log
|
+---models
|       recommenderCF.py
|       recommenderMF.py
|
+---results
|
+---scripts
|       dataset.py
|       grid_search_update.py
|       requirements_script.py
|       tabulator_script.py
|
\---__pycache__
        (Compiled Python files)
```

## 🚀 Getting Started
### 1️⃣ Install Dependencies
Make sure you have Python installed, then run:
```sh
pip install -r requirements.txt
```

### 2️⃣ Run the Main Program
```sh
python main.py
```

## 📊 Datasets
This project utilizes the MovieLens datasets:
- **ml-100k**: Smaller dataset for quick testing.
- **ml-1m**: Larger dataset for more robust testing.

## 🔍 Evaluation
The evaluation scripts assess the accuracy of different recommendation algorithms:
- `evaluator.py`: Main evaluation logic.
- `surprise_evaluation.py`: Evaluation for the `Surprise` library.

## 🏗 Models
Two main recommendation models:
- `recommenderCF.py`: Collaborative Filtering implementation.
- `recommenderMF.py`: Matrix Factorization-based recommender.

## 📜 Logging
Logging is stored in `logs/dataset.log` for tracking dataset operations.

## 📄 Scripts
- `dataset.py`: Handles dataset loading and preprocessing.
- `grid_search_update.py`: Performs hyperparameter tuning.
- `requirements_script.py`: Manages package dependencies.
- `tabulator_script.py`: Formats or structures data.

## 📌 Notes
- **Results**: Output files are stored in the `results/` directory.
- **Cache Files**: `__pycache__/` stores compiled Python files.

## 💡 Future Improvements
- Implement more advanced recommendation algorithms.
- Optimize dataset processing speed.
- Add a web-based interface for easy interaction.

## 📜 License
This project is open-source. Feel free to modify and improve it!


