
# CG10 – Meta-Fusion Ensemble of Transformer Models for Robust Multilingual and Cross-Domain Sentiment Classification


## Team Info
- 22471A05I3 — **Prasanna Panchumarthi** ( [LinkedIn](https://www.linkedin.com/in/prasanna-panchumarthi/) )
**_Work Done:** Complete project ownership including problem formulation, dataset selection, preprocessing, transformer model training (BERT, RoBERTa, DistilBERT, XLM-RoBERTa), meta-fusion ensemble design using MLP, experimentation, evaluation, result analysis, and research paper writing.

- 22471A05G6 — **Jaya Lakshmi Kshatriya** ( [LinkedIn](https://www.linkedin.com/in/jaya-lakshmi) )
**_Work Done:** Assisted in experimentation review and documentation support.

- 22471A05I9 — **Apsa Shaik** ( [LinkedIn](https://www.linkedin.com/in/apsa-appu-shaik-361a69287/) )
**_Work Done:** Assisted in literature survey support and result verification.

---

## Abstract

Sentiment classification has progressed from simple polarity detection to multilingual and cross-domain applications, yet domain shifts and linguistic variability remain major challenges to robust generalization. This paper presents a meta-fusion ensemble framework that integrates four transformer models—BERT, RoBERTa, DistilBERT, and XLM-RoBERTa—each fine-tuned on benchmark datasets including Sentiment140, IMDB, ABSA, and a multilingual corpus. Unlike static ensemble approaches such as majority voting or averaging, the proposed method employs a trainable Multi-Layer Perceptron (MLP) to dynamically fuse model logits, effectively capturing inter-model dependencies. Experiments conducted on a balanced evaluation set of 2,700 samples across diverse domains and languages demonstrate the effectiveness of this framework. The meta-fusion ensemble achieved an accuracy of 86.91% and a macro-F1 score of 85.67%, outperforming both individual transformer baselines and static fusion methods. These results confirm the advantage of learnable ensemble strategies for improving sentiment prediction under domain and language variability.

---

## Paper Reference (Inspiration)
👉 **[Paper Title Sentiment Analysis in the Era of Large Language Models
  – Author Name Zhang et al
 ] 

---

## Our Improvement Over Existing Paper

* Introduced a trainable meta-fusion layer instead of static voting or averaging
* Performed logit-level fusion to capture inter-model dependencies
* Evaluated across multiple domains and multilingual datasets
* Achieved improved neutral sentiment classification, which is typically challenging
* Demonstrated better generalization and robustness compared to single-model approaches

---

## About the Project

This project builds a robust sentiment analysis system capable of handling text from multiple domains and languages.

**Workflow:**

* Input text (tweets, reviews, multilingual content)
* Text preprocessing and tokenization
* Independent sentiment prediction using transformer models
* Dynamic logit fusion using MLP
* Final sentiment output (Positive / Negative / Neutral)
The system is designed to overcome domain shift and linguistic variability.

---

## Dataset Used
👉 **Multiple Benchmark Sentiment Datasets**

**Dataset Details:**

* Sentiment140: Twitter sentiment (binary classification)
* IMDB Reviews: Movie review sentiment (binary classification)
* SemEval-2014 ABSA: Aspect-based sentiment (3-class)
* Multilingual Dataset: Sentiment data across 10+ languages
* Total Samples: 2,700 (balanced across datasets)

---

## Dependencies Used

Python, PyTorch, HuggingFace Transformers, PyABSA, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

---

## EDA & Preprocessing

* Removal of URLs, special characters, emojis, and noise
* Text normalization and lowercasing
* Tokenization using model-specific tokenizers
* Padding and truncation to fixed sequence length
* Balanced sampling across datasets to avoid bias

---

## Model Training Info

* Base Models: BERT, RoBERTa, DistilBERT, XLM-RoBERTa
* Optimizer: Adam
* Learning Rate: 2e-5
* Epochs: 2–4
* Batch Size: 16
* Loss Function: Cross-Entropy Loss
* Fusion Model: Multi-Layer Perceptron (MLP) with ReLU activation

---

## Model Testing / Evaluation

* Accuracy
* Precision
* Recall
* Macro and Weighted F1-score
* Confusion Matrix
* Training vs Validation performance analysis

---

## Results

* **Meta-Fusion Ensemble Accuracy:** 86.91%
* **Macro-F1 Score:** 85.67%
* Outperformed individual transformer models and traditional ML baselines
* Significant improvement in neutral sentiment prediction

---

## Limitations 

* Higher computational cost due to multiple transformer models
* Limited evaluation on very low-resource languages

## Future Work

* Zero-shot and cross-lingual evaluation
* Multimodal sentiment analysis
* Attention-based fusion strategies
* Real-time deployment optimization

---

## Deployment Info

* Trained and tested on Google Colab (GPU-enabled)
* Models saved for inference and reuse
* Can be deployed using Flask/FastAPI or Hugging Face Spaces

---
