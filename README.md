# 🚀 Government Data Q&A Chatbot

🌐 **Live App:** [https://governmentdata-chatbot.streamlit.app/](https://governmentdata-chatbot.streamlit.app/)

---

## 🗂️ Overview

A conversational AI dashboard to query **Indian government datasets** (weather & agriculture) in natural language.  
Smartly fetches and engineers data, then answers complex analytical questions with tables and insights.

---

## 🏗️ Backend Workflow

### 1️⃣ Data Fetching & Feature Engineering

- **🔑 DataGovIndia API:**  
  Secure connection using your API key from data.gov.in.
- **📚 Metadata Sync:**  
  Syncs resource IDs, titles, available fields, time ranges, and scope.
- **🎛️ Feature Extraction:**  
  Extracts and standardizes entities (year, state, crop, rainfall, production) from metadata and dataset titles.
- **🛠️ Data Cleaning:**  
  Cleans dataframe columns, handles NA values, creates aggregates and derived features for analysis.

---

### 2️⃣ Chatbot Query Pipeline

- **🗨️ Query Understanding:**  
  - User submits any question (“Compare wheat output in UP and Bihar since 2010, and show rainfall impact”).
  - LLM extracts entities (state, crop, year), intent (comparison, top N, correlation), and routes to relevant sources.
- **🧠 Intent Classification & Entity Recognition:**  
  Plans for advanced **NLP models** (spaCy, transformers) to boost query intent & entity extraction accuracy.
- **🔍 Dataset Filtering:**  
  Intelligent filtering ensures selected datasets fully match query entities and intent—checks fields before fetching.
- **🔗 Parameter Building:**  
  Builds precise API filters for live data retrieval, limiting records for speed and cost.

---

### 3️⃣ Data Retrieval & Analyst Answer

- **⚡ Fetches filtered data using DataGovIndia API.**
- **🔎 Feature engineering prepares data for analysis (e.g., pivoting, aggregation).**
- **🤖 AI summaries:**  
  LLM generates easy-to-read answers, including state/crop/district breakdowns, trends, and citations.

---


---

## 💻 Usage

1. Visit [the app](https://governmentdata-chatbot.streamlit.app/)
2. Type a natural-language question  
3. Select suggested datasets if prompted
4. Receive summarized, cited answer + download raw CSV

---

## ✨ Features

- AI-powered multi-entity filtering and matching
- Handles multi-state, multi-year, and complex queries
- Backend feature engineering for reliable analysis
- Download-ready, clean dataframes
- Secure: API keys/secrets **never exposed** publicly

---

## 🌱 Future Improvements

- **🚀 Advanced NLP for query intent & entity recognition:**  
  Integrate spaCy or transformer-based models for richer parsing.
- **⚡ Embeddings & semantic search:**  
  Use vector similarity to recommend datasets even with non-matching field names.
- **🔄 Auto data joins:**  
  Correlate and join weather/crop/production datasets for composite analysis.
- **📊 Visualization API:**  
  Generate interactive charts and graphs alongside text answers.
- **🗃️ New dataset integrations:**  
  Plug in additional government domains—health, population, infrastructure, etc.
- **💬 User feedback & auto-correction:**  
  Improve filtering and matching via feedback loop.
- **🧩 Modular pipeline:**  
  Swappable backend modules for easy extension by contributors.

---

## 🤝 Contributing

Ideas welcome—fork, raise PRs, or open issues.  
**Please do not ever commit API keys or secrets!**  
See `.gitignore` and setup notes for best practices.

---

**Made with ❤️ for open-data research and analytics.**


