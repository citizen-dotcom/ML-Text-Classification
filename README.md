# Text_Classification
Text Classification with Traditional ML and Deep Learning Approaches
1. Introduction
This report is a complete study of text classification which is here using conventional machine learning and deep learning techniques. The major aim of this project is to study the applicability of certain classification systems across various, in this case I have chosen 3, datasets to find out the best approaches suitable for text classification tasks. It investigates the varying performance of models with respect to different feature extraction methods employed and also examines whether there are any significant advantages of using deep learning models over traditional machine learning methods. The project is in line with the increasing necessity of automatic text classification in applications ranging from sentiment classification to spam detection.

2. Methodology
2.1 Text Preprocessing
Subjecting the text to machine learning, the following steps are included in the pre-processing pipeline which help in model training as well as the overall accuracy in this case: First, converting all text to lowercase so that it is consistently worked upon. Then, using regular expressions to remove all punctuation marks, numbers, and special characters from the text. Next, tokenizing the text and then removing stopwords that contribute little to the discriminative value. Finally, applying lemmatization to reduce words to their base forms, which generally pegs the vocabulary size and improves the quality of features. This standard preprocessing setup was applied uniformly across all data to enable a fair comparison between the results.

2.2 Feature Extraction
Two main features were used for extraction; Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF). The BoW model uses a vector representation of text based on word counts, yielding a sparse yet high-dimensional representation of the text. On the other hand, TF-IDF refines this technique by using weights that emphasize term presence in the document and its inverse presence across the corpus-the more distinctive the term, the greater its weight. For both approaches, the vocabulary was limited to the top 5,000 features in order to avoid having too many dimensions while preserving discriminative power.

2.3 Model Selection
We employed four traditional machine-learning algorithms: Multinomial Naive Bayes, Logistic Regression, Linear Support Vector Machine (SVM), and Random Forest. Effectively, the choice of the models is based on their proven efficiency in performing text classification tasks. In addition, we have deep learning methods based on the use of Long Short-Term Memory (LSTM) networks with an incorporated attention mechanism. This LSTM model makes use of word embeddings to capture the semantic relationships among words and sequential information that most traditional models miss.

2.4 Hyperparameter Tuning
In the optimal hyperparameter selection process for traditional models, GridSearchCV along with 3-fold cross-validation were used. Parameters that were tuned include regularization strength for Logistic Regression and SVM, alpha values for Naive Bayes, and tree depth and estimator count for Random Forest. We put into practice early stopping and learning rate scheduling to avert overfitting and enhance convergence capabilities of the LSTM model.

3. Datasets
3.1 IMDb Movie Reviews
The IMDb dataset is comprised of 10,000 movie reviews which are well balanced between the positive and negative sentiment classes. The average text length is nearly 237 words per review and the vocabulary counts over 159,000 unique words. It is the classic case of sentiment analysis concerning long documents and rich vocabulary.

3.2 Twitter Sentiment Analysis
Like the IMDb dataset, the Twitter sentiment dataset was comprised of 10,000 tweets classified as either positive or negative. Although in a different domain, it proved to have distribution statistics similar to those already mentioned: average of 232 words per tweet and a vocabulary size of about 159,000 words. The shorter, less formal nature of tweets presents quite different challenges compared with movie reviews.

3.3 SMS Spam Collection
The SMS spam dataset consists of 5,574 legitimate and spam messages. Spam messages of this crudely structured dataset contribute to only 13.4% of the dataset. SMS messages are much shorter than movie reviews or tweets, averaging just 15.6 words, with a much smaller vocabulary comprising 15,733 unique words. This dataset represents a challenge in class imbalance and a case of short-text classification.

4. Results
4.1 Overall Performance Comparison
The experiments here that we conduct found that SMS Spam dataset has shown superior classification performance in all models as it gave F1 scores even above 0.97 consistently. The IMDb and Twitter datasets were tougher, since the F1 scores from these ranged between 0.81 and 0.85. Logistic Regression and Multinomial Naive Bayes proved to be more successful among the rest of the classical methods on all of these datasets. The deep learning LSTM approach yielded competitive results achieving F1 scores of 0.819, 0.822, and 0.983 over the IMDb, Twitter, and SMS datasets respectively. An extrememly important thing to understand is that the data quality is as important as the code.

4.2 Feature Extraction Impact
In terms of the IMDb dataset, the classi-fier technique of BoW performed relatively better with logistic regression (F1 score 0.851) than TF-IDF (F1 score 0.830); in the case of the Twitter dataset, the same trend was observed. However, with regard to the SMS Spam dataset, the Multinomial Naive Bayes used TF-IDF features better and thus, gave the highest overall F1 score of 0.983. This indicates that the best method of feature extraction might depend on the dataset, where TF-IDF may act favorably in the short text domain with distinct discriminative terms.

4.3 Model-Specific Performance
Logistic Regression tends to yield superior results across datasets, particularly those characterized by BoW features; it gains the highest F1 scores on these datasets: 0.851 on IMDb and 0.853 on Twitter. Multinomial Naive Bayes performed better than the others on the SMS dataset, achieving an F1 score of 0.983 using the TF-IDF features. Linear SVM was strong in performance but usually ranked lower than Logistic Regression. Random Forest was consistently the worst performer of all the models, indicating that tree ensembles may not be the most suitable algorithms to use with the highly dimensionality nature of text data.

4.4 Deep Learning Comparison
The LSTM model equipped with an attention mechanism exhibited F1 scores of 0.819 (IMDb), 0.822 (Twitter), and 0.983 (SMS), similar to those of the best traditional models. While deep learning on these datasets did not stand out as distinctly outperforming traditional methods, it demonstrated consistent performance without requiring substantial feature engineering, thus hinting at potential advantages on more complex text classification tasks.

5. Analysis and Conclusions
5.1 Key Findings
One of the experiments we conducted was to test whether ordinary machine learning models might still be important. Logistic Regression and Multinomial Naive Bayes were very useful to us for text classification when used judiciously according to the required feature extraction methods. Generally, choice of input, BoW or TF-IDF, should be dependent on the data. BoW does well with long input and TF-IDF does better on short text, highly domain-specific. They have not outperformed ordinary methods significantly in these data sets despite being more complicated. In fact, some cases suggest marginal improvements with them as well.
