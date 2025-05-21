# Zee Personalized Movie Recommender System

A Python-based project implementing a comprehensive movie recommendation engine using collaborative filtering (item-based and user-based), cosine similarity, Pearson correlation, and matrix factorization. This repository demonstrates end-to-end data handling, exploratory analysis, model building, and evaluation.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Repository Structure](#repository-structure)
* [Data Files](#data-files)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
* [Modeling Approach](#modeling-approach)
* [Evaluation Metrics](#evaluation-metrics)
* [Questionnaire & Insights](#questionnaire--insights)
* [Contributing](#contributing)
* [License](#license)

---

## Project Overview

This repository contains a full-featured movie recommender system built on the MovieLens dataset. It covers:

* Data ingestion, cleaning, and feature engineering
* Exploratory data analysis (EDA) and visualizations
* Item-based collaborative filtering using Pearson correlation and cosine similarity
* User-based collaborative filtering
* Matrix factorization via the `Surprise` library
* Model evaluation (RMSE, MAPE)
* Embedding visualization

The goal is to deliver personalized movie recommendations for a target user by leveraging ratings from similar users and similar items.

---

## Repository Structure

```text
.
├── Business Case_ Zee Recommender Systems Approach.pdf  # Project requirements and approach overview
├── Details.docx                                      # Additional project details
├── Zee_Recommender_Systems.ipynb                     # Jupyter notebook with full implementation
├── zee-movies.dat                                    # Movie metadata (MovieID, Title, Genres)
├── zee-ratings.dat                                   # User ratings (UserID, MovieID, Rating, Timestamp)
└── zee-users.dat                                     # User demographics (UserID, Gender, Age, Occupation, Zip-code)
```

---

## Data Files

* **`zee-movies.dat`** — contains `MovieID::Title::Genres`
* **`zee-ratings.dat`** — contains `UserID::MovieID::Rating::Timestamp`
* **`zee-users.dat`** — contains `UserID::Gender::Age::Occupation::Zip-code`

Each file uses `::` as the delimiter. These are read and merged within the notebook.

---

## Prerequisites

* Python 3.7 or higher
* Jupyter Notebook

Required Python libraries:

```bash
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
surprise               # for matrix factorization
```

---

## Installation

1. **Clone the repository**

   ```bash
   ```

git clone [https://github.com/GopalGB/Zee-Personalized-Recommender-System.git](https://github.com/GopalGB/Zee-Personalized-Recommender-System.git)
cd Zee-Personalized-Recommender-System

````
2. **Create and activate a virtual environment** (optional but recommended)
   ```bash
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate  # Windows
````

3. **Install dependencies**

   ```bash
   ```

pip install -r requirements.txt

````

> *Note: You can also install each library individually if a `requirements.txt` is not provided.*

---

## Usage

Open and run the notebook to reproduce the analysis and model building steps:

```bash
jupyter notebook Zee_Recommender_Systems.ipynb
````

Key sections in the notebook:

1. **Data Loading & Merging** — read `.dat` files, parse, and merge into a single DataFrame.
2. **Exploratory Data Analysis** — summary statistics, rating distributions, average ratings by genre.
3. **Feature Engineering** — extract release year, compute number of ratings per movie/user.
4. **Collaborative Filtering**

   * *Item-based:* Pearson correlation, cosine similarity, KNN
   * *User-based:* Pearson correlation scoring for new user ratings
5. **Matrix Factorization** — train and evaluate SVD model using `surprise`.
6. **Embedding Visualization** — plot 2D embeddings of movies and users.

---

## Modeling Approach

1. **Item-based Collaborative Filtering**

   * Construct movie-user pivot table
   * Compute pairwise item similarity via Pearson correlation
   * Recommend top 5 similar movies for a given title
2. **Cosine Similarity & KNN**

   * Generate item and user similarity matrices
   * Use `scikit-learn`’s `NearestNeighbors` for fast lookups
3. **Matrix Factorization**

   * Apply SVD (d=4 latent factors) with the `surprise` library
   * Train/test split and cross-validation
   * Tune hyperparameters and record RMSE/MAPE
4. **User-based Collaborative Filtering (Optional)**

   * Prompt for new user ratings
   * Identify top-K similar users via Pearson correlation
   * Aggregate weighted ratings to recommend top 10 movies

---

## Evaluation Metrics

* **Root Mean Squared Error (RMSE)** — measures prediction accuracy
* **Mean Absolute Percentage Error (MAPE)** — percentage-based error metric

Results and plots are available in the notebook under the "Model Evaluation" section.

---

## Questionnaire & Insights

The notebook also answers these questions:

1. **Age group** with the most ratings
2. **Profession** with the highest engagement
3. **Gender distribution** among raters
4. **Decade** with the most movie releases
5. **Top-rated movies** and most-rated movie
6. **Similar movies** to ‘Liar Liar’ (item-based)
7. **CF classification**: user-based vs. item-based
8. **Similarity ranges**: Pearson (\[−1, +1]) vs. Cosine (\[0, 1])
9. **Matrix factorization performance** (RMSE & MAPE)
10. **Sparse matrix representation example**

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, enhancements, or new features.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

*Happy recommending!*
