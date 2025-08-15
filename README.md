# AI Summarization (BERT NER → Gemini)
This is a school project I made :). It aims to solve the chunking data loss problem of GenAI models by using an BERT NER model to provide context, then using Gemini to chunk the document using the given context. 
This solution aims to minimise the amount of data lost when chunking long documents into summaries :).

## Requirements
1. `Python 3.11`
2. `pip`
3. `Google Gemini API Key`

## Quickstart
1. Create and activate a venv:
   `py -3.11 -m venv .venv`
   `.\.venv\Scripts\Activate.ps1`
2. Install Dependencies
   `pip install -r requirements.txt`
3. Add env with the following variables (you can get the model zip url from GitHub releases in this repository by going to ner_bert_gmb.zip):
   `GEMINI_API_KEY` `MODEL_ZIP_URL`
4. Run web application (side note: you can use main.py to debug the backend by using `python main.py`):
   `uvicorn app.app:app --reload`
