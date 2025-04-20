# بسم الله الرحمن الرحيم
# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report
# Download only once :)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')  # Add this to fix the error


################################################################################
## 1. Reviews preprocessing: remove stop words and apply stemming using NLTK. ##
################################################################################
## 2.Labels (positive, negative, neutral) mapping into numerical labels.      ##
################################################################################
negation_words = {'no', 'not', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere'}
df = pd.read_csv('amazon_reviews.csv')
stop_words = set(stopwords.words('english')) - negation_words

stemmer = SnowballStemmer('english')


label_map = {
    'positive': 2,
    'neutral': 1,
    'negative': 0
}

inv_label_map = {v: k for k, v in label_map.items()} 

df['sentiments'] = df['sentiments'].map(label_map)

print("Stemming, Labeling, and filtering stop words ..")
def preprocess_text(text):
    words = word_tokenize(str(text))
    filtered_words = [w for w in words if w.lower() not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)

df['cleaned_review'] = df['cleaned_review'].apply(preprocess_text)
df.to_csv('filtered_reviews.csv', index=False)
print("Preprocessing finished")

#########################################################
## 3. Data splitting into 80% training and 20% testing.##
#########################################################

print("training ...")
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_review'], df['sentiments'], test_size=0.2, random_state=42
)
print("training finished")


#########################################################
## 4. TF-IDF Vectorization                           ##
#########################################################


print("Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(
    ngram_range=(1,3),
    min_df=2,       
    max_df=0.9,
    max_features=15000
    
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"TF-IDF Train shape: {X_train_tfidf.shape}")
print(f"TF-IDF Test shape: {X_test_tfidf.shape}")

print("TF-IDF vectorization finished")

#########################################################
## 5. SVM Model                         ##
#########################################################


print("Training SVM...")
svm = SVC(class_weight='balanced')
svm.fit(X_train_tfidf, y_train)
y_pred_svm = svm.predict(X_test_tfidf)
print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm,zero_division=0))


##########################################################
## 6. Logistic Regression Model                         ##
##########################################################

print("Training Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000,) 
log_reg.fit(X_train_tfidf, y_train)
y_pred_log_reg = log_reg.predict(X_test_tfidf)
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg,zero_division=0))

#########################################################
## 7. Naive Bayes Model                               ##
#########################################################
print("Training Naive Bayes...")
nb = ComplementNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)
print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb,zero_division=0))


#########################################################
## 8. User input for prediction.                     ##
#########################################################
print("User input for prediction...")
while True:
    user_input = input("Your review (type 'exit' to quit): ")
    if user_input.strip().lower() == 'exit':
        break
    cleaned = preprocess_text(user_input)
    user_tfidf = vectorizer.transform([cleaned])

    pred_svm = inv_label_map[ svm.predict(user_tfidf)[0] ]
    pred_log_reg = inv_label_map[ log_reg.predict(user_tfidf)[0] ]
    pred_nb = inv_label_map[ nb.predict(user_tfidf)[0] ]
    print("SVM Prediction: ",pred_svm)
    print("Logistic Regression Prediction: ",pred_log_reg)
    print("Naive Bayes Prediction: ",pred_nb)

print("Predictions finished")

