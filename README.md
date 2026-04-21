# NLP Summer Term 2026

Welcome to the NLP Class of 2026. A general overview of the repository:
- [assignments](./assignments/): subfolders with weekly assignments as Markdown file and marimo notebook as template
- [data](./data): our lyrics datasets
- [notebooks](./notebooks/): tutorial and example notebooks that are used in class
- [slides](./slides/): contains lecture slides and the exam rubrics 

## Schedule

| **#** | **Date** | **CW** | **Weekday** | **Time** | **Room** | **Topic** | **Contents** | **Slides** |
|-------|----------|--------|-------------|----------|----------|-----------|--------------|-----------|
| 1 | 16.04. | 16 | Thursday | 15:15–18:30 | 149 | Introduction: The World of NLP | What is NLP? Typical NLP tasks.  | [PDF](./slides/01_kick_off.pdf) |
| 2 | 23.04. | 17 | Thursday | 15:15–18:30 | 149 | Text Processing Fundamentals | Regex, tokenization (rule-based, subword), text normalization, stemming & lemmatization, stop words | [PDF](./slides/02_text_processing.pdf) |
| 3 | 30.04. | 18 | Thursday | 15:15–18:30 | 149 | Linguistic Annotations & Pattern Matching | POS tagging, dependency parsing, NER, spaCy Matcher, PhraseMatcher, EntityRuler | [PDF](./slides/03_pattern_matching.pdf) |
| 4 | **05.05.** | 19 | **Tuesday** | 15:15–18:30 | **154** | ⚠️ **Substitute date for 14.05.**  Text Vectorization: From Text to Numbers | Bag-of-Words, document-term matrices, TF-IDF, cosine similarity, TfidfVectorizer | |
| 5 | 07.05. | 19 | Thursday | 15:15–18:30 | 149 | Text Classification | Naive Bayes, Logistic Regression, evaluation (precision, recall, F1), cross-validation, imbalanced data | |
| 6 | 28.05. | 22 | Thursday | 15:15–18:30 | 149 | Language Models, Word Vectors + **🎤 Guest Lecture Ines Montani** | **1st half (90 min):** From sparse to dense: why BoW/TF-IDF isn't enough. What is a language model? Word2Vec & GloVe  **2nd half (90 min): 🎤 Guest Lecture Ines Montani** | |
| - | 01.06. | 23 | Monday | | 390 | **📝 Master DS Exams (afternoon)** | | |
| 7 | **02.06.** | 23 | **Tuesday** | 15:15–18:30 | **154** | ⚠️ **Substitute date for 04.06.** BERT: Pre-Training & Fine-Tuning | From static to contextualized embeddings. What is pre-training? Masked language modeling. BERT for text classification. Fine-tuning with Hugging Face | |
| 8 | 11.06. | 24 | Thursday | 15:15–18:30 | 149 | The BERT Ecosystem | SentenceBERT & sentence embeddings, BERTopic for topic modeling, semantic search. When to fine-tune vs. use off-the-shelf | |
| 9 | 18.06. | 25 | Thursday | 15:15–18:30 | 149 | Introduction to LLMs | What are LLMs? Evolution. Capabilities, API access, basic prompting, few-shot learning | |
| 10 | 25.06. | 26 | Thursday | 15:15–18:30 | 149 | Working with LLMs | Chain-of-thought, spaCy-LLM integration, LLMs for NLP tasks (classification, NER, summarization). When to use what: classical vs. BERT vs. LLM decision framework | |
| 11 | 02.07. | 27 | Thursday | 15:15–18:30 | 149 | Ethics, Limitations & Exam Preparation | Hallucination, bias, ethics in NLP | |
| - | 09.07. | 28 | Thursday | | 400 | **📝 Exams (afternoon)** | **Oral group exams** | |
| - | 10.07. | 28 | Friday | | 400 | **📝 Exams (afternoon)** | **Oral group exams** | |

## Initial Set-Up

- Packages are managed in the [pyproject.toml](./pyproject.toml)
- Install `uv` on your machine. Then run

```bash
uv sync
```

## Exam

### Bachelor DIS
- The exams are in groups of three (i.e. your team) on the 9th and 10th of July in the afternoon
- There are three broad categories text processing & statistics (sessions 2 - 4), text classification (sessions 4 - 8), and LLMs (sessions 9 - 11) 
- You distribute the categories among the team
- Each team member starts with presenting one completely new approach within their category which was not discussed during the lecture and exercises. For example
    - A new statistic and visualization
    - A new classification (including the evaluation)
    - A new use of LLMs
- The presentation takes around 5 to 7 minutes per team member including follow-up questions
- After each presentation, the team member is then asked two random questions from a question pool about their implementation of the exercises in the other two categories. The team member has to show that they understand the NLP theory behind this implementation.
    - so, make sure that your marimo notebooks are up and running

[Scoring Rubric](./slides/rubric_dis.pdf)

### Master DS

- On the 22nd of May you will receive a question
    - "Imagine your task is XXX. How would you try to solve it? What are possible challenges?"
- On the 29th of May submit 2 pages answering the question via mail
- Oral exam
    - 7 minutes presentation of the answers, max 6 slides
    - 5 minutes questions about the presentation
    - 7 minutes questions about the NLP lecture

[Scoring Rubric](./slides/rubric_ds.pdf)

## Assignments

- Submit your work on the main branch in the respective assignemnt folder
- Set a new upstream to fetch updates on our main repo
    - `git remote add upstream https://github.com/th-koeln-iws-nlp/nlp-ss2026.git` 
    - `git pull upstream main`
- One assignment per week as issue
- Once you close the issue (or I close it for you) a code review agent will give you feedback on your code
- You can then re-open the issue and implement the feedback and close it again. The agent will run again.
- If you have questions, create an issue in your repo and tag me `@RichardSiegTH`


## Dataset

- We mainly work with the [Billboard Top 100 Lyrics Dataset](https://www.kaggle.com/datasets/brianblakely/top-100-songs-and-lyrics-from-1959-to-2019)
- Use the [550k dataset](https://www.kaggle.com/datasets/serkantysz/550k-spotify-songs-audio-lyrics-and-genres) if you want to get more recent songs and more songs by one artist
- Lyrics were scraped from Genius
- Is already cleaned (some lyrics where also completely removed)
- Most of the songs are enriched with data from the Spotify API 

The available fields and explanations can be found in the [data folder](./data/available_fields.md).

## Literature & Links

### Textbooks and Course Materials

- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) - Comprehensive NLP textbook by Dan Jurafsky and James H. Martin, covering language models, transformers, text classification, speech recognition, and more (free online)
- [Introduction to Natural Language Processing](https://mitpress.mit.edu/9780262042840/introduction-to-natural-language-processing/) - Technical perspective on NLP methods by Jacob Eisenstein, synthesizing classical representations with contemporary machine learning techniques
- [COS 484: Natural Language Processing](https://princeton-nlp.github.io/cos484/) - Princeton course on modern NLP covering language models, transformers, machine translation, and LLM systems
- [CMPT 825: Natural Language Processing](https://anoopsarkar.github.io/nlp-class/syllabus.html) - Simon Fraser University course by Anoop Sarkar focusing on contemporary approaches including LLMs, attention mechanisms, and model optimization

### Interactive Online Courses

- [Advanced NLP with spaCy](https://github.com/explosion/spacy-course) - Free interactive course covering spaCy fundamentals, rule-based and neural approaches, with exercises in multiple languages
- [NLP Course](https://lena-voita.github.io/nlp_course.html) - Interactive lecture-blogs by Lena Voita with visualizations, analysis, and interpretability sections on embeddings, transformers, and transfer learning
- [A Hands-on Introduction to Natural Language Processing](https://github.com/deskool/nlp-class) - YouTube lecture series with Jupyter notebooks by Prof. Mohammad Ghassemi covering fundamentals, neural models, transformers, and speech processing

### Resource Collections

- [Awesome NLP](https://github.com/keon/awesome-nlp) - Curated list of 18k+ starred resources including libraries, datasets, courses, and multilingual NLP tools across 40+ languages
- [Text Pattern Matching Guide](https://codecut.ai/regex-pregex-pyparsing-text-pattern-matching-guide/) - Practical tutorial comparing regex, pregex, and pyparsing for text extraction and pattern matching tasks