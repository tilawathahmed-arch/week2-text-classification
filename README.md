# Week 2: Designing and Implementing Text Classification Strategies

A standalone NLP project that builds and evaluates supervised sentiment-classification pipelines for short movie-review snippets using the Cornell Sentence Polarity Dataset v1.0.

## Project contents

- `REPORT.md` — strategy, methodology, model comparison, validation plan, metrics, limitations, and future improvements.
- `train_classifier.py` — executable end-to-end training and evaluation pipeline.
- `requirements.txt` — required Python packages.
- `submission_description.txt` — internship portal description (200+ words).

## Dataset

The project uses the **Cornell Sentence Polarity Dataset v1.0**: 10,662 Rotten Tomatoes snippets, comprising 5,331 positive and 5,331 negative examples. This is the sentence-level v1.0 corpus, not the separate document-level movie-review polarity dataset v2.0.

Official source: https://www.cs.cornell.edu/people/pabo/movie-review-data/

The script first attempts to download the official Cornell archive. If that fails, it tries a public plain-text mirror of the same v1.0 files. You can also download the files manually and place them in the data directory.

## Methodology

The task is binary sentiment classification. Candidate models are a most-frequent-class dummy baseline, CountVectorizer with Multinomial Naive Bayes, TF-IDF with Multinomial Naive Bayes, TF-IDF with Logistic Regression, and TF-IDF with LinearSVC.

The workflow uses a stratified 80/20 train-test split and stratified five-fold cross-validation on the training partition. Macro-F1 is the primary model-selection metric; mean accuracy breaks ties. Vectorizers are inside scikit-learn Pipelines, so vocabulary and IDF statistics are fitted separately within each training fold. The final test partition is kept out of model selection.

## Run locally

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python train_classifier.py --download --data_dir data --out_dir outputs
```

To use local files instead of downloading, place `rt-polarity.pos` and `rt-polarity.neg` (or the mirror's equivalent files) under `data/`, then omit `--download`.

## Generated outputs

The script writes cross-validation comparisons, held-out test metrics, test predictions, a confusion-matrix plot, a cross-validation model-comparison plot, top feature weights, a fitted model, and a prepared dataset to the output directory.

**Performance note:** the full 10,662-example model run has not been executed as part of preparing this repository. The script was smoke-tested on a small synthetic fixture only; those fixture scores are not Cornell dataset results. Run the script on the complete public corpus to calculate actual cross-validation and test metrics.

## Dataset and research references

- Pang, B. and Lee, L. (2005). *Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales*. Proceedings of ACL.
- Cornell University, Movie Review Data: https://www.cs.cornell.edu/people/pabo/movie-review-data/
- scikit-learn text feature extraction: https://scikit-learn.org/stable/modules/feature_extraction.html
- scikit-learn Pipeline documentation: https://scikit-learn.org/stable/modules/compose.html
