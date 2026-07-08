# Email Spam Detection

## Project Overview

This project builds an email spam detection system using Natural Language Processing (NLP) and machine learning. The goal is to classify email/SMS text messages as either **spam** or **ham** using text preprocessing, feature extraction, and classification models.

The project compares multiple NLP feature extraction techniques and evaluates their performance using classification metrics such as precision, recall, F1 score, accuracy, and confusion matrix.

## Dataset

The dataset used is `spam.csv`, which contains labeled text messages.

Main columns:

- `v1`: Target label, either `ham` or `spam`
- `v2`: Email/message text

Unnecessary unnamed columns are removed during preprocessing, and the remaining columns are renamed as:

- `label`
- `text`

## Project Workflow

1. Import required libraries
2. Load the dataset
3. Clean and prepare the data
4. Perform exploratory data analysis
5. Preprocess text data
6. Convert labels into numerical format
7. Split data into training and testing sets
8. Apply feature extraction techniques
9. Train machine learning models
10. Evaluate model performance
11. Compare results and select the best model

## Text Preprocessing

The text preprocessing pipeline includes:

- Lowercasing text
- Removing numbers, punctuation, and special characters
- Tokenizing words
- Removing English stopwords
- Lemmatizing words
- Joining cleaned tokens back into text

This helps reduce noise and improves the quality of input features for machine learning models.

## Feature Extraction Techniques

The following feature extraction methods are used:

### 1. Bag of Words

Bag of Words converts text into a matrix of word counts. It represents how frequently each word appears in each message.

### 2. TF-IDF

TF-IDF stands for Term Frequency-Inverse Document Frequency. It gives higher importance to words that are meaningful in a message but not too common across all messages.

### 3. Word2Vec

Word2Vec creates dense vector representations of words. In this project, a Word2Vec model is trained on the available dataset.

### 4. Pretrained Word2Vec

The project also uses pretrained Google News Word2Vec embeddings to compare performance with locally trained embeddings.

## Models Used

The main machine learning model used is:

- Logistic Regression

Logistic Regression is trained separately using each feature extraction method.

## Evaluation Metrics

The models are evaluated using:

- Confusion Matrix
- Precision Score
- Recall Score
- F1 Score
- Accuracy Score

For spam detection, **recall and F1 score are especially important** because missing spam messages can reduce the usefulness of the model.

## Final Result

Based on model comparison, **Bag of Words with Logistic Regression performed the best** for this dataset.

The local Word2Vec model did not perform well because the dataset is relatively small, so it could not learn strong word embeddings. Pretrained Word2Vec performed better than local Word2Vec, but it still did not outperform Bag of Words.

## Conclusion

This project successfully demonstrates how NLP techniques can be used to detect spam messages. Traditional feature extraction methods like Bag of Words and TF-IDF work well on this dataset, especially when combined with Logistic Regression.

The best final model for this project is:

```text
Bag of Words + Logistic Regression
```

This model provides the best balance of accuracy, precision, recall, and F1 score among the tested approaches.

## How to Run

1. Open the notebook:

```text
Email_Spam_Detection.ipynb
```

2. Make sure the dataset is in the same folder:

```text
spam.csv
```

3. Install the required libraries if needed:

```bash
pip install pandas numpy nltk scikit-learn matplotlib gensim
```

4. Download NLTK resources if needed:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

5. Run all cells from top to bottom.

## Project Files

```text
Email_Spam_Detection.ipynb   Main Jupyter notebook
spam.csv                     Dataset
Email Spam Detection.pdf     Project PDF/report
README.md                    Project documentation
```

## Future Scope

This project can be improved further by adding:

- Hyperparameter tuning using GridSearchCV
- More models such as Naive Bayes, SVM, Random Forest, and XGBoost
- A model comparison table
- ROC-AUC and PR-AUC evaluation
- Threshold tuning to improve spam recall
- A function to predict whether a new email is spam or ham
- Model saving using Pickle or Joblib
- Deployment using Streamlit or Flask
- Larger and more recent spam email datasets
- Advanced NLP models such as BERT, DistilBERT, or sentence-transformers

## Author
### Imtiyaj Ali Shaikh

This project was created as part of an NLP learning project for email spam detection.
