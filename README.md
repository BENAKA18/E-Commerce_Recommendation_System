🎯 E-Commerce Recommendation System






📌 Overview

This project implements an E-Commerce Recommendation System using multiple machine learning techniques.
The objective is to recommend relevant products to users based on popularity, user interactions, and product similarity. The project also includes a comparison between PCA and SVD for dimensionality reduction.

An interactive web interface is developed using Streamlit to explore and compare recommendations.

🧠 Problem Statement

E-commerce platforms face challenges such as cold-start users, sparse rating data, and scalability.
This project addresses these issues by implementing different recommendation strategies and evaluating dimensionality reduction techniques to improve recommendation quality.

📁 Project Structure
├── app.py
├── product-recommendation-system-for-e-commerce.ipynb
├── requirements.txt
├── README.md
└── data/
    ├── product_descriptions.csv
    └── ratings_Beauty.csv

🚀 Getting Started
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Application
streamlit run app.py

🎯 Recommendation Models Implemented
Model	Description	Best Use Case
🔥 Popularity-Based	Recommends most popular products	New users
👥 Collaborative Filtering	Uses user–item interactions	Personalized recommendations
🔍 Content-Based	Uses product descriptions	Cold-start products
📊 PCA vs SVD Analysis

The project compares:

Principal Component Analysis (PCA)

Singular Value Decomposition (SVD)

Evaluation Metrics:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Explained Variance

Visualizations are used to analyze performance differences.

🛠️ Technologies Used

Python 3.8+

Streamlit

Pandas

NumPy

Scikit-learn

Plotly

📘 Jupyter Notebook

The notebook includes:

Data exploration and preprocessing

Recommendation model development

PCA vs SVD performance comparison

Visual analysis

📈 How to Use the Application

Load the datasets using the sidebar

Select a recommendation model

Generate product recommendations

Compare PCA and SVD using adjustable components

🐞 Troubleshooting

Ensure required datasets are available

Install all dependencies from requirements.txt

Reduce dataset size if performance issues occur

📄 License

MIT License

👤 Author

Benaka
