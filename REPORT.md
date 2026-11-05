# Designing and Implementing Text Classification Strategies
## Week 2 Internship Task — Natural Language Processing

**Prepared by:** Shaik Tilawath Ahmed  
**Institution:** Geethanjali  
**Dataset:** Cornell Sentence Polarity Dataset v1.0

## Executive Summary

This report defines a practical, leakage-aware strategy for classifying short movie-review snippets as positive or negative. It uses the same Cornell sentence polarity dataset v1.0 as the exploratory analysis task so that data preparation and classification remain consistent. The implementation compares a count-based Naive Bayes baseline, TF-IDF with Naive Bayes, TF-IDF with Logistic Regression, and TF-IDF with a linear Support Vector Machine (LinearSVC), alongside a most-frequent-class dummy baseline.

Model selection uses stratified five-fold cross-validation on the training partition, with macro-F1 as the primary selection metric and mean accuracy as a tie-breaker. A separate stratified test partition is reserved for one final evaluation after selection. The accompanying Python script executes the workflow and exports comparison tables, test metrics, predictions, visualizations, and model artifacts when run on the complete corpus.

**Scope and honesty note:** this deliverable is a strategy report plus an executable training/evaluation script. The full 10,662-record model run was not executed while preparing this report. No classifier accuracy or winning model is claimed here. The script is designed to calculate those values reproducibly on the actual corpus.

## 1. Problem Statement and Objectives

Given a short English-language movie-review snippet, predict whether its sentiment label is positive or negative. This is a supervised binary text-classification problem. The target label is taken from the source dataset and is not inferred from the text during training.

Objectives:
1. Establish a simple, reproducible baseline using traditional machine-learning methods.
2. Compare count-based and TF-IDF feature representations with Naive Bayes, Logistic Regression, and LinearSVC.
3. Use validation that protects the final test set and prevents vocabulary/IDF leakage.
4. Evaluate aggregate performance and class-specific errors, then identify practical next steps.

The pipeline is considered complete when it loads and validates both classes, creates a reproducible stratified split, compares candidate models using training-only cross-validation, evaluates the selected model once on the held-out test set, and saves the results.

## 2. Dataset Description

The project uses the **Cornell Sentence Polarity Dataset v1.0**, which contains 10,662 Rotten Tomatoes snippets: 5,331 positive and 5,331 negative. Each line contains one short English-language text snippet. This is the sentence-level v1.0 corpus, not Cornell's separate document-level movie-review polarity dataset v2.0.

| Property | Description |
|---|---|
| Source | Cornell Movie Review Data, sentence polarity dataset v1.0 |
| Total examples | 10,662 snippets |
| Positive class | 5,331 (50%) |
| Negative class | 5,331 (50%) |
| Input | One English-language text snippet per line |
| Target | Positive or negative sentiment |
| Known limitation | Snippet labels are inherited from review-level Fresh/Rotten ratings and may contain label noise |

The official dataset page is https://www.cs.cornell.edu/people/pabo/movie-review-data/.

## 3. Text Preparation

The script loads both source files, assigns labels from the file names, creates stable snippet identifiers, and removes blank records. It applies conservative normalization: lowercasing, URL and email replacement, removal of markup tags, filtering of unusual characters, and whitespace normalization.

Negation, contractions, and useful punctuation are retained. A phrase such as “not good” can have the opposite meaning from “good”; removing negators as stopwords can therefore damage sentiment information. Word bigrams are included to capture some local phrase context.

## 4. Feature-Extraction Strategies

| Method | Representation | Strengths | Trade-offs |
|---|---|---|---|
| Bag of Words / CountVectorizer | Token counts, optionally with n-grams | Simple, interpretable baseline | Common terms may dominate; no semantic understanding |
| TF-IDF | Term frequency weighted by inverse document frequency | Emphasizes terms distinctive to a snippet | Sparse lexical features; limited semantic understanding |
| Word and character n-grams | Word unigrams/bigrams or character fragments | Bigrams capture phrases and negation; character features can help with spelling variation | More features and sparsity; parameters need tuning |
| Pretrained embeddings / transformers | Dense contextual representations | Can capture semantic similarity and context | Greater computational and implementation complexity |

The initial classical experiment uses word unigram and bigram features. CountVectorizer supports the Naive Bayes count baseline; TF-IDF is used for the other linear models.

## 5. Candidate Algorithms

| Algorithm | Role | Advantages | Limitations |
|---|---|---|---|
| Multinomial Naive Bayes | Fast baseline on count and TF-IDF features | Simple and efficient for sparse text | Conditional-independence assumption; probabilities may be poorly calibrated |
| Logistic Regression | Linear discriminative model on TF-IDF | Regularization, interpretable coefficients, probability output | Requires regularization tuning; linear decision boundary |
| LinearSVC | Margin-based linear classifier on TF-IDF | Strong high-dimensional sparse-text baseline | No native probability estimates; decision scores are not calibrated probabilities |
| Dummy most-frequent classifier | Reference baseline | Shows performance from predicting only the majority class | Does not learn language patterns |

## 6. Training and Validation Strategy

1. Load and validate both positive and negative source files.
2. Normalize snippets while preserving negation and contractions.
3. Create a stratified 80/20 train-test split using a fixed random seed.
4. Build candidate vectorizer-classifier Pipelines.
5. Run stratified five-fold cross-validation on the 80% training partition.
6. Compare candidates using mean macro-F1; use mean accuracy as a tie-breaker and retain fold standard deviations.
7. Refit the selected complete Pipeline on all training examples.
8. Predict the untouched 20% test partition once.
9. Review false positives and false negatives, including errors involving negation, contrast, short snippets, sarcasm, and possible label noise.
10. Save comparison tables, metrics, predictions, plots, top feature weights, and the fitted model.

A scikit-learn Pipeline ensures that the vectorizer and its vocabulary/IDF statistics are fitted only within each training fold during cross-validation. The holdout test partition is not used for model selection. Any later hyperparameter search should remain inside a training-only validation procedure.

## 7. Evaluation Metrics

| Metric | Interpretation | Purpose |
|---|---|---|
| Accuracy | Fraction of all predictions that are correct | Overall performance |
| Precision | Fraction of predicted positives that are truly positive | Reliability of positive predictions |
| Recall | Fraction of actual positives correctly identified | Coverage of positive examples |
| F1-score | Harmonic mean of precision and recall | Balance of precision and recall |
| Macro-F1 | Mean of positive- and negative-class F1 | Primary model-selection metric; equal class weight |
| Weighted-F1 | Class F1 values weighted by class support | Support-weighted summary |
| ROC-AUC | Ranking quality from decision scores or probabilities | Threshold-independent comparison when scores are available |

A classification report and confusion matrix provide class-specific information. For the confusion matrix, rows represent true labels and columns represent predicted labels; the positive class is the positive sentiment label.

## 8. Challenges and Mitigation

| Challenge | Why it matters | Mitigation |
|---|---|---|
| Label noise | A snippet may not express the overall review's rating | Inspect representative errors and describe results as dataset-specific |
| Negation and contrast | “Not good” and “good, but predictable” require phrase context | Preserve negators and contractions; include bigrams |
| Short snippets | Limited context can make polarity ambiguous | Review class-wise errors; consider contextual representations later |
| Vocabulary sparsity | Names and rare terms create many low-frequency features | Use minimum document frequency, regularization, and compare word/character n-grams |
| Data leakage | Fitting IDF before splitting can inflate scores | Keep vectorization and classification in one Pipeline inside cross-validation |
| Domain and time limits | Older movie snippets may not generalize to social posts or current reviews | Validate on a separate target-domain corpus |
| Metric imbalance | Accuracy can hide poor performance on one class | Report macro and per-class metrics plus a confusion matrix |

## 9. Future Enhancements

- Tune vectorizer settings and model regularization through a documented training-only grid search.
- Compare unigram-only, unigram-plus-bigram, binary count, and character n-gram representations.
- Inspect high-weight terms and group errors to identify shortcuts or dataset-specific artifacts.
- Evaluate calibrated probabilities and decision thresholds where downstream use requires confidence scores.
- Experiment with word embeddings or a compact transformer after the classical baselines establish a reliable reference point.
- Test generalization on a newer movie-review or product-review dataset after checking its license and domain suitability.

## 10. Reproducibility and Implementation Deliverables

The companion `train_classifier.py` script implements corpus retrieval, validation, preprocessing, vectorizer-classifier Pipelines, stratified cross-validation, final holdout evaluation, error export, and artifact saving. The run metadata records the UTC run date, Python version, scikit-learn version, random seed, holdout size, and fold count.

Run:

```bash
pip install -r requirements.txt
python train_classifier.py --download --data_dir data --out_dir outputs
```

Expected outputs include `cross_validation_comparison.csv`, `test_metrics.json`, `test_predictions.csv`, `confusion_matrix.png`, `cv_model_comparison.png`, `top_model_features.csv`, `best_model.joblib`, and `prepared_dataset.csv`.

A small synthetic fixture was used only to smoke-test the script interface and output generation. Its scores are not dataset results and are intentionally excluded from the report. Run the script against the full Cornell dataset to obtain the actual cross-validation and holdout metrics.

## 11. Conclusion

This strategy defines a feasible classical NLP classification experiment for balanced positive/negative movie-review snippets. It compares count and TF-IDF representations, evaluates Naive Bayes, Logistic Regression, and LinearSVC, and uses stratified cross-validation plus an untouched holdout test set. Macro-F1, class-wise metrics, and confusion-matrix inspection provide a more informative evaluation than accuracy alone. The accompanying script turns the plan into a reproducible implementation while leaving full-corpus performance to be measured on the actual data.

## References

1. Cornell University. Movie Review Data: sentence polarity dataset v1.0. https://www.cs.cornell.edu/people/pabo/movie-review-data/
2. Pang, B. and Lee, L. (2005). *Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales*. Proceedings of ACL.
3. scikit-learn documentation: text feature extraction. https://scikit-learn.org/stable/modules/feature_extraction.html
4. scikit-learn documentation: Pipeline and composite estimators. https://scikit-learn.org/stable/modules/compose.html
