# CrediTrust Financial Complaint-Answering Chatbot

## Overview

This project develops an internal AI tool for **CrediTrust Financial**, a digital finance company in East Africa, to analyze customer complaints from the **Consumer Financial Protection Bureau (CFPB)** dataset (February 2014–August 2015). The tool empowers product, support, and compliance teams to:

- Identify complaint trends
- Reduce analysis time from days to minutes
- Enable proactive issue resolution across five product categories:
  - Credit card
  - Personal loan
  - Buy Now, Pay Later (BNPL)
  - Savings account
  - Money transfers

The project consists of four major tasks:

1. **Data Preprocessing and Exploratory Data Analysis (EDA)**
2. **Text Chunking, Embedding, and FAISS Vector Store Creation**
3. **Retrieval-Augmented Generation (RAG) Pipeline**
4. **Gradio-based Interactive Interface**

The solution is implemented in **Python** using **Google Colab** with GPU acceleration, and outputs are stored in **Google Drive**.

---

## 🛠️ Prerequisites

- **Google Colab**: Free tier with GPU (T4 recommended) or Colab Pro for longer sessions
- **Google Drive**: Required to persist input/output files (~3-4 GB space needed)
- **Install Python Libraries**:

```bash
pip install pandas numpy matplotlib seaborn langchain \
sentence-transformers faiss-cpu transformers torch gradio

