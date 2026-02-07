**Datasets Used**

This project leverages four benchmark datasets to evaluate the robustness, generalization, and multilingual capabilities of the proposed Meta-Fusion Ensemble of Transformer Models.
Each dataset addresses a distinct sentiment analysis challenge, enabling cross-domain and cross-lingual evaluation.

1️⃣ **Sentiment140 Dataset**

**Dataset Title**
**Sentiment140** – Twitter Sentiment Analysis Dataset

**Usage of Dataset**

Used to evaluate model robustness on noisy, short-text social media data containing slang, emojis, misspellings, and informal grammar.

**Dataset Information**

**Dataset Name**: Sentiment140

**Source**: Kaggle

**Domain**: Social Media (Twitter)

**Task**: Sentiment Classification

**Problem Type**: Binary Classification

**File Format**: CSV

**Dataset Link**:
https://www.kaggle.com/datasets/kazanova/sentiment140

**Dataset Overview**

**Total Records**: ~1,600,000 tweets

**Labeled Records**: Fully labeled

**Classes**: Positive, Negative

**Annotation Type**: Automatically labeled using emoticons

**Why This Dataset?**

Simulates real-world noisy data

Tests model robustness against informal text

Essential for validating sentiment performance on social platforms

**Features Used**

Tweet Text

Sentiment Label

2️⃣ **IMDB Movie Reviews Dataset**

**Dataset Title**
**IMDB Dataset of 50K Movie Reviews**

**Usage of Dataset**

Used for document-level sentiment classification on long-form, structured text.

**Dataset Information**

**Dataset Name**: IMDB Reviews

**Source**: Kaggle

**Domain**: Movie Reviews

**Task**: Sentiment Classification

**Problem Type**: Binary Classification

**File Format**: CSV

**Dataset Link**:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

**Dataset Overview**

**Total Records**: 50,000 reviews

**Labeled Records**: Fully labeled

**Classes**: Positive, Negative

**Annotation Type**: Human-annotated

**Why This Dataset?**

Evaluates transformer performance on long documents

Tests contextual understanding and semantic coherence

Complements short-text datasets like Twitter

**Features Used**

Review Text

Sentiment Label

3️⃣ **Multilingual Tweets Dataset**

**Dataset Title**
**Multilingual and Code-Mixed Tweets Dataset**

**Usage of Dataset**

Used to evaluate multilingual and Hindi–English code-mixed sentiment classification, a major real-world challenge.

**Dataset Information**

**Dataset Name**: Multilingual Tweets Dataset

**Source**: Kaggle

**Domain**: Social Media

**Task**: Multilingual Sentiment Analysis

**Problem Type**: Multi-Class Classification

**File Format**: CSV

**Dataset Link**:
https://www.kaggle.com/datasets/suraj520/multi-task-learning

**Dataset Overview**

**Total Records**: ~14,000 tweets

**Labeled Records**: Fully labeled

**Classes**: Positive, Negative, Neutral

**Annotation Type**: Human-annotated

**Why This Dataset?**

Handles low-resource and code-mixed languages

Tests cross-lingual generalization

Strengthens novelty of the proposed ensemble approach

**Features Used**

Tweet Text

Language Tags

Sentiment Label

4️⃣**SemEval 2014 Aspect-Based Sentiment Dataset**

**Dataset Title**
**SemEval 2014 Task 4 – Aspect-Based Sentiment Analysis**

**Usage of Dataset**

Used for aspect-level sentiment classification across multiple domains (Restaurants and Laptops).

**Dataset Information**

**Dataset Name**: SemEval 2014 ABSA

**Source**: Kaggle

**Domain**: Reviews (Restaurants & Laptops)

**Task**: Aspect-Based Sentiment Analysis

**Problem Type**: Multi-Class Classification

**File Format**: XML (Converted to CSV)

**Dataset Link:**
https://www.kaggle.com/datasets/charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis

**Dataset Overview**

**Total Records**: ~6,000 aspect-level instances

**Labeled Records**: Fully labeled

**Classes**: Positive, Negative, Neutral

**Annotation Type**: Human-annotated

**Why This Dataset?**

Enables fine-grained sentiment analysis

Supports domain-specific sentiment reasoning

Crucial for validating aspect-aware transformer models

**Features Used**

Review Text

Aspect Term

Aspect Category

Sentiment Polarity

**Summary**

The selected datasets collectively enable:

Cross-domain evaluation (social media, reviews)

Multilingual and code-mixed sentiment analysis

Aspect-level sentiment reasoning

Robust ensemble validation across diverse data distributions

This comprehensive dataset selection strengthens the generalization, novelty, and publication-worthiness of the proposed Meta-Fusion Ensemble Framework.

