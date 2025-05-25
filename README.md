# Reviews Sentiment Classification 🔍
A natural language processing project that classifies Amazon product reviews into Positive, Neutral, or Negative sentiments using natural language processing techniques and TF-IDF vectorization.

## Objectives 🎯
This project aims to build a sentiment classifier that:

* **Takes User Input:**
Accepts a product review text and predicts its sentiment.

* **Core Functionalities:**

    **Preprocessing**: Cleans the text by removing stop words and applying stemming

    **Label Mapping**: Converts text labels to numerical values
  
    **Data Splitting**: Trains models on 80% of the data and tests on 20%

    **Vectorization**: Transforms text into numerical features using TF-IDF

    **Model Training**: Trains three classifiers:
        Support Vector Machine (SVM)
        Logistic Regression
        Naive Bayes
  
* **Output:**
  **Prediction:** Predicts the sentiment of a new review entered by the user

## Technologies Used 🛠️

* Python
* NLTK – Text preprocessing (stop word removal, stemming)
* Git & GitHub

## Running the Project 🚀
1. clone the respository
```bash
git clone https://github.com/Reviews-Sentiment-Classification
```

2. Download Dataset Ensure amazon_reviews.csv is in the root directory. This file contains 17,000+ labeled Amazon product reviews.
3.  Run the Program
Open the .ipynb file in Jupyter Notebook or run the script file:

```bash
python ID1_ID2_ID3_ID4.py
```

## Sample Input & Output 📊
1. Example 1
``` bash
Review: "This product is amazing and exceeded my expectations!"  
Predicted Sentiment: Positive
```

2. Example 2
```bash
Review: "It's okay, not too bad but not great either."  
Predicted Sentiment: Neutral
```

## Contributors 🤝
- Mazen Ahmoudadly
- Salah Eddin
- Abdelrhman Ezzat

