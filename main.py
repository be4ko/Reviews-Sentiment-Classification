# بسم الله الرحمن الرحيم
# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download only once :)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')  # Add this to fix the error


################################################################################
## 1. Reviews preprocessing: remove stop words and apply stemming using NLTK. ##
################################################################################
## 2.Labels (positive, negative, neutral) mapping into numerical labels.      ##
################################################################################

df = pd.read_csv('amazon_reviews.csv')
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')


label_map = {
    'positive': 2,
    'neutral': 1,
    'negative': 0
}
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
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"TF-IDF Train shape: {X_train_tfidf.shape}")
print(f"TF-IDF Test shape: {X_test_tfidf.shape}")

idf = vectorizer.idf_
feature_names = vectorizer.get_feature_names_out()  
idf_dict = dict(zip(feature_names, idf))
print("\nIDF Values:")
print(idf_dict)


