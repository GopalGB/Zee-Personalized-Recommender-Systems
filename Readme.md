# Zee Personalized Recommender System

This repository contains the implementation of a personalized movie recommender system built using collaborative filtering and matrix factorization techniques on the MovieLens dataset. The goal is to provide accurate, user-centric movie recommendations by analyzing user ratings and finding patterns in both item-based and user-based approaches.

## 🚀 Features

* **Data Processing**: Read and clean raw `ratings.dat`, `users.dat`, and `movies.dat` files.
* **Exploratory Data Analysis**: Statistics on ratings distribution, user demographics, and movie genres.
* **Item-based Collaborative Filtering**:

  * Pearson Correlation Coefficient
  * Cosine Similarity
  * K-Nearest Neighbors (using scikit-learn) for item similarity
* **User-based Collaborative Filtering** (optional):

  * Similarity computation between users
  * Weighted recommendation based on top-n similar users
* **Matrix Factorization**:

  * Factorization using the Surprise library
  * Evaluation with RMSE and MAPE
  * Embedding visualization (2D and 4D)

## 📁 Repository Structure

```
Zee-Personalized-Recommender-System/
│
├── data/
│   ├── ratings.dat       # UserID::MovieID::Rating::Timestamp
│   ├── users.dat         # UserID::Gender::Age::Occupation::Zip-code
│   └── movies.dat        # MovieID::Title::Genres
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_item_based_CF.ipynb
│   ├── 04_user_based_CF.ipynb        # Optional
│   ├── 05_matrix_factorization.ipynb
│   └── 06_embeddings_visualization.ipynb
│
├── src/
│   ├── data_loader.py                # Functions to read & merge data files
│   ├── preprocessing.py              # Cleaning and feature engineering
│   ├── cf_item.py                    # Item-based CF (Pearson & Cosine)
│   ├── cf_user.py                    # User-based CF methods
│   ├── mf_model.py                   # Matrix factorization implementation
│   └── utils.py                      # Helper functions
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project introduction and instructions
└── LICENSE                           # Open-source license
```

## 🛠️ Prerequisites

* Python 3.7+
* pandas
* numpy
* scikit-learn
* scipy
* matplotlib / seaborn
* Surprise (for matrix factorization)

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🔧 Usage

1. **Clone the repository**

   ```bash
   ```

git clone [https://github.com/GopalGB/Zee-Personalized-Recommender-System.git](https://github.com/GopalGB/Zee-Personalized-Recommender-System.git)
cd Zee-Personalized-Recommender-System

````

2. **Download and place data**
   - Obtain `ratings.dat`, `movies.dat`, `users.dat` from the [MovieLens dataset](https://drive.google.com/drive/folders/1RY4RG7rVfY8-0uGeOPWqWzNIuf-iosuv)
   - Place files into the `data/` directory

3. **Run Jupyter Notebooks**
   ```bash
jupyter notebook
````

* Follow the numbered notebooks in `notebooks/` for step-by-step execution.

4. **Generate recommendations**

   * Use `cf_item.py` to recommend top-5 similar movies for a given title.
   * Use `mf_model.py` to train matrix factorization and evaluate performance.

## 📊 Methodology

1. **Data Preparation & EDA**

   * Merge ratings, movies, and users into a single DataFrame
   * Extract release year from movie titles
   * Compute rating counts and average ratings per movie

2. **Item-based Collaborative Filtering**

   * Build a user–item pivot table
   * Compute Pearson correlation and cosine similarity between item vectors
   * For a target movie, find top-5 neighbors by similarity

3. **User-based Collaborative Filtering** (optional)

   * Prompt new user ratings
   * Find similar users via Pearson correlation
   * Aggregate weighted ratings to recommend movies

4. **Matrix Factorization**

   * Use Surprise’s SVD algorithm with latent dimension d=4
   * Train/test split and evaluate RMSE & MAPE
   * Extract user and item embeddings
   * Visualize embeddings for d=2

## 🎯 Evaluation Metrics

* **RMSE** (Root Mean Squared Error)
* **MAPE** (Mean Absolute Percentage Error)

Record your scores in the notebooks under the Matrix Factorization section.

## 📈 Embeddings Visualization

* Generate and compare 2D projections of item/item and user/user embeddings
* Analyze clustering patterns relative to genres or demographics

## 📋 Questionnaire & Insights

* Which age group watches the most movies?
* Top occupations by rating count
* True/False: Most raters are male
* Most common movie release decade
* Movie with maximum ratings
* Top-3 recommendations for "Liar Liar"
* CF categories: Item-based vs User-based
* Similarity ranges: Pearson (–1 to +1), Cosine (0 to 1)

## 🤝 Contributing

Contributions are welcome! Please fork the repo and open a pull request with:

* New similarity algorithms
* Performance optimizations
* Additional evaluation metrics

## 📄 License

This project is open-sourced under the MIT License. See [LICENSE](LICENSE) for details.

---

Happy recommending!
