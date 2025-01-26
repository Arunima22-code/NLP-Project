#Roadmap

1. **Setting Up Development Environment**
2. **Generating or Collect Synthetic Data**
3. **Data Cleaning & Preprocessing**
4. **Data Augmentation**
5. **Multi-Label Text Classification**
6. **Entity Extraction (Domain Knowledge Base + NER)**
7. **Summarization**
8. **Building a REST API (or CLI)**
9. **Containerization (Docker)**
10. **Reporting & Final Submission**

---

## 1. Setting Up Development Environment

### Why This Step?
We need a consistent place to run and share code. This helps us to avoid environment/version conflicts (e.g., differing library versions). It also makes it easier for others to replicate your work and ensures that your final Docker container will run correctly.

### How To Do It
1. **Choosing a coding environment**:
   - **Google Colab** (very common for quick AI experimentation; free GPU/TPU time).
   - **Local environment** (PyCharm, VSCode, etc.) if you prefer everything on your own machine.
   - **Virtual environment / Conda environment** to isolate your dependencies.

2. **Installing the required Python libraries**:
   - `pandas`, `numpy` for data manipulation.
   - `scikit-learn` or `pytorch`/`tensorflow` for classification models.
   - `spaCy` or `transformers` (Hugging Face) for advanced NLP.
     

3. **Version Control**:
   - Create a **GitHub repository** (public, as requested).
   - Ensuring to commit your code notebooks, Python scripts, and Dockerfiles.


## 2. Generating or Collecting Synthetic Data

### Why This Step?

How To Do It
1. **Create your CSV** (`calls_dataset.csv`):
   - Fields:
     1. **id**: A unique identifier (e.g., 1, 2, 3…).
     2. **text_snippet**: A short piece of text (e.g., “We love the analytics, but CompetitorX has a cheaper subscription.”).
     3. **labels**: A comma-separated list of labels (e.g., “Positive, Pricing Discussion, Competition”).
   - Make sure you have a variety of text (at least 100 rows) to represent different scenarios: some that talk about pricing, some that mention competitors, some about product features, objections, compliance, security, etc.

2. **Creating the domain knowledge JSON** (`domain_knowledge.json`):
   - Example structure:
     ```json
     {
       "competitors": ["CompetitorX", "CompetitorY", "CompetitorZ"],
       "features": ["analytics", "AI engine", "data pipeline"],
       "pricing_keywords": ["discount", "renewal cost", "budget", "pricing model"]
     }
     ```
   - This dictionary will help for simple “string matching” to detect domain-specific entities.



## Step 3: Data Cleaning & Preprocessing

1. **Loading the Dataset**  
   - **What**: reading our `calls_dataset.csv` into a DataFrame.  
   - **Why**: We need a structured data format (DataFrame) to easily manipulate and inspect the text.

2. **Removing Duplicates**  
   - **What**: We dropped any repeated rows.  
   - **Why**: Duplicate data can lead to overfitting (the model sees the same text multiple times), and it inflates our dataset without real new information.

3. **Text Normalization**  
   - **What**: We converted the text to lowercase, removed punctuation, stripped extra spaces, and optionally removed digits.  
   - **Why**: Standardizing text (e.g., “Hello” vs. “hello”) helps the model treat words consistently. Removing punctuation often reduces noise in bag-of-words methods.

4. **Tokenization & Lemmatization**  
   - **What**: We split each sentence into individual words (tokens), then converted each word to its root form (lemmatizing “running” → “run”).  
   - **Why**: Tokenization is needed so the model can work with each word. Lemmatization helps group variations of the same word (e.g., “runs,” “running,” “ran”) so they’re treated as one term.

5. **Splitting Labels**  
   - **What**: We turned comma-separated label strings (e.g., “Pricing Discussion, Security”) into Python lists (e.g., `["Pricing Discussion", "Security"]`).  
   - **Why**: Multi-label classification frameworks need labels in a list form to handle multiple tags per snippet.

6. **Saving the Cleaned Dataset**  
   - **What**: We wrote the final output to `calls_dataset_cleaned.csv`.  
   - **Why**: Re-saving ensures all processing steps are done once, and we can reuse the cleaned file without repeating the cleaning each time.


### Documentation for Step 4: Data Augmentation


#### **What is Data Augmentation?**
Data augmentation involves creating new, slightly modified versions of existing data to improve the diversity of the dataset. In this case, we applied augmentation techniques to text snippets in the dataset to handle potential class imbalances and enhance the performance of the machine learning model.

---

#### **Why Did We Use It?**
1. **Address Class Imbalance**: Some labels may have fewer examples in the dataset, causing the model to perform poorly on those labels. Augmentation balances this.
2. **Improve Model Generalization**: By introducing variations, the model learns better and avoids overfitting.
3. **Enrich Dataset Variety**: It helps simulate real-world variability, making the dataset more robust.

---

#### **How Did We Implement It?**
We used the following augmentation techniques on each row of the dataset:

1. **Synonym Replacement**: Replace random words with synonyms (e.g., "discount" → "rebate").
2. **Random Insertion**: Insert new random words from a predefined dictionary (e.g., adding "performance" in the sentence).
3. **Random Deletion**: Randomly delete words from the text while keeping the sentence meaningful.
4. **Random Swap**: Swap the positions of two random words in the sentence.

For each row in the original dataset, **4 new augmented rows** were generated, resulting in a dataset 5 times larger than the original.



#### **How Is This Helpful?**
1. **Balanced Dataset**: Rare labels like "Objection" or "Security" now have more examples, reducing model bias toward common labels.
2. **Enhanced Model Performance**: The model sees a variety of text structures and learns better patterns, improving accuracy and robustness.
3. **Scalable**: Even with a small dataset, augmentation creates enough data for training.

---

#### **Example**

**Original Text**:
> "We are impressed with the analytics engine, but CompetitorA has a cheaper pricing model."

**Augmented Versions**:
1. **Synonym Replacement**:
   > "We are amazed with the analytics engine, but CompetitorA has a cheaper pricing model."
2. **Random Insertion**:
   > "We are impressed with the analytics engine, but CompetitorA pricing has a cheaper pricing model."
3. **Random Deletion**:
   > "We are with analytics engine, but CompetitorA has pricing model."
4. **Random Swap**:
   > "We are analytics impressed with the engine, but CompetitorA has a cheaper pricing model."

---

#### **Final Dataset Summary**
- **Original Rows**: 150
- **Augmented Rows**: 600
- **Total Rows**: 750
- **Output File**: `augmented_calls_dataset.csv`


---
### Documentation: Step 5 - Multi-Label Text Classification

---

#### **Objective**
To train a machine learning model that can assign multiple labels (categories) to a single text snippet from the dataset. Each snippet in the dataset can belong to one or more categories, such as "Pricing Discussion," "Competition," or "Positive."

---

#### **What We Did**
1. **Loaded the Dataset**:
   - Used the augmented dataset `augmented_calls_dataset.csv` created in Step 4.
   - Each row in the dataset contains:
     - `text_snippet`: A sales/marketing-related text.
     - `labels`: Comma-separated labels indicating the snippet's categories.

2. **Preprocessed the Data**:
   - Converted comma-separated labels into **lists** (e.g., `"Positive, Pricing Discussion"` → `["Positive", "Pricing Discussion"]`).
   - Transformed text snippets into numerical features using **TF-IDF Vectorization**:
     - Extracted the top 5000 unique terms from the text (ignoring stop words).
     - Represented each snippet as a vector based on term importance (TF-IDF score).

3. **Multi-Hot Encoding for Labels**:
   - Used `MultiLabelBinarizer` to convert label lists into **multi-hot vectors**:
     - Example: 
       - Labels: `["Positive", "Pricing Discussion"]`
       - Multi-hot Vector: `[1, 0, 0, 1, 0, 0]`  
         (if the full set of labels is: `["Positive", "Negative", "Objection", "Pricing Discussion", "Security", "Competition"]`).

4. **Split the Data**:
   - Divided the dataset into:
     - **80% training data**: Used to train the model.
     - **20% test data**: Used to evaluate the model's performance.

5. **Trained the Model**:
   - Chose **Logistic Regression** with `OneVsRestClassifier` for multi-label classification:
     - Each label is treated as a separate binary classification problem.
     - The model learns to predict whether each label applies to a given snippet.

6. **Evaluated the Model**:
   - Measured performance using:
     - **Precision**: How many of the predicted labels were correct.
     - **Recall**: How many of the actual labels were correctly predicted.
     - **F1-Score**: A balanced measure of Precision and Recall.
   - Printed a **classification report** showing these metrics for each label.

7. **Saved the Model and Preprocessing Steps**:
   - Saved the trained model, vectorizer, and label encoder for future use in building an inference pipeline.

---

#### **Why This Step is Beneficial for the Assessment**

1. **Core Objective**: 
   - The assignment explicitly requires a multi-label classification system. This step fulfills that requirement by building a model that can assign multiple categories to a snippet.

2. **Improved Accuracy**:
   - By splitting the dataset and evaluating the model, we ensure that the system generalizes well and is not just memorizing the training data.

3. **Reusable Pipeline**:
   - Saving the model, vectorizer, and label encoder ensures that this pipeline can be reused in the next steps (e.g., for building a REST API).

4. **Demonstrates Data Science Expertise**:
   - Handling multi-label problems is more challenging than single-label classification. Successfully implementing this step demonstrates advanced understanding of machine learning.

---

#### **Example**

1. **Input Snippet**:
   - "We love the analytics, but CompetitorX has a cheaper subscription."

2. **True Labels**:
   - `["Positive", "Pricing Discussion", "Competition"]`

3. **TF-IDF Representation**:
   - The text snippet is converted into a vector (e.g., `[0.1, 0.0, 0.5, 0.2, ...]`) based on the importance of terms.

4. **Model Prediction**:
   - The model outputs probabilities for each label:
     - Positive: 0.92 → **1 (Predicted)**
     - Negative: 0.12 → 0
     - Objection: 0.05 → 0
     - Pricing Discussion: 0.87 → **1 (Predicted)**
     - Security: 0.08 → 0
     - Competition: 0.79 → **1 (Predicted)**

5. **Final Output**:
   - Predicted Labels: `["Positive", "Pricing Discussion", "Competition"]`

---

#### **How It Helps in Real Scenarios**
- This trained model can automatically analyze sales/marketing call snippets and categorize them into relevant buckets. This is useful for companies to:
  - Understand common customer concerns (e.g., pricing, security).
  - Identify competitors mentioned in conversations.
  - Track positive and negative feedback trends.

---

## 6. Entity Extraction (Domain Knowledge Base + NER)

### Why This Step?
We  need to detect domain-specific keywords (like competitor names or discount mentions) but also be able to handle more general entity recognition (for example, “SOC2,” “ISO compliance,” or anything not in your domain dictionary).

### How To Do It
1. **Dictionary Lookup**:
   - Load `domain_knowledge.json`.
   - For each text snippet, do a simple substring match to find any mention of the words in `competitors`, `features`, `pricing_keywords`.
   - Collect these matches into a set or list of “domain-based entities.”

2. **NER / Advanced Extraction**:
   - **Regex** or rule-based approach: Search for patterns. For example, if you often get references to certifications like “SOC2,” “ISO 27001,” you can have a small custom regex for these patterns.
   - **Pre-trained NER** (spaCy or Hugging Face):
     - spaCy approach: `doc = nlp(text_snippet)`, then check `ent.text` and `ent.label_`.
     - Combine them with your domain-based extraction. 
     - Example:
       ```python
       spacy_nlp = spacy.load("en_core_web_sm")
       doc = spacy_nlp(text_snippet)
       general_entities = [(ent.text, ent.label_) for ent in doc.ents]
       ```

3. **Merge Domain + NER Entities**:
   - Combine the dictionary-based entities with the general NER. 
   - Potentially storing them as:
     ```json
     {
       "snippet_id": 1,
       "extracted_entities": {
         "domain": ["CompetitorX", "discount"],
         "general_ner": [("SOC2", "ORG")]
       }
     }
     ```


## 7. Summarization


**Implementation**:
   - Quick solution with Hugging Face:
     ```python
     from transformers import pipeline
     summarizer = pipeline("summarization", model="t5-small") 
     summary = summarizer(text_snippet, max_length=30, min_length=5, do_sample=False)
     print(summary[0]['summary_text'])
     ```



