# QA System

A lightweight LLM-powered question-answering system that responds to natural-language questions about member data collected from Aurora’s public `/messages` API. The system interprets queries such as 
_“When is Layla planning her trip to London?”_ or _“How many cars does Vikram Desai have?”_ by retrieving relevant messages and generating concise answers using GPT. The service is deployed and publicly 
accessible here:  https://qa-system-dqgp.onrender.com/docs

## Project Structure
.
├── app/
│ ├── main.py # FastAPI service exposing /ask endpoint
│ ├── qa.py # Retrieval + LLM answering pipeline
│ ├── analysis.py # Dataset anomaly detection (bonus requirement)
│ └── models.py # Response schema models
├── requirements.txt
├── Dockerfile
└── README.md

## Tech Stack
Python  
FastAPI  
OpenAI 
Uvicorn  
RapidFuzz  
Docker + Render for deployment

## Usage

### Clone the Repository
```bash
git clone https://github.com/Veda0718/QA_System.git
cd QA_System
```
### Create a virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Configure Environment Variables
Create a .env file in the project root:
```bash
OPENAI_API_KEY=your_key_here
```
### Run the Server Locally
```bash
uvicorn app.main:app --reload
```
After running, open: http://localhost:8000/docs
## Docker Usage
### Build Image
```bash
docker build -t aurora-qa .
```
### Run Container
```bash
docker run -e OPENAI_API_KEY=your_key_here -p 8000:8000 aurora-qa
```
Then open: http://localhost:8000/docs

## Anomaly Detection
The project includes `analysis.py`, which performs automated anomaly checks on the concierge message dataset.  To run locally:
```bash
python -m app.analysis
```
This script detects:
 - Duplicate messages
 - Missing required fields (user, text, timestamp)
 - Very short or empty messages
 - Impossible or future timestamps
 - Burst messages from the same user
 - Underspecified intent requests (via GPT when API key is configured)
Currently, the script analyzes the latest 300 messages fetched from the backend.

## Alternative Approaches Considered

 1. RapidFuzz for semantic ranking (initial prototype) — Initially used RapidFuzz for top-K similarity search across messages because it’s fast and offline-friendly, but it only handles lexical similarity, so we replaced it with GPT-based semantic reasoning for more accurate intent understanding and context retrieval.
 2. RAG with embeddings — Could store all messages in a vector database (FAISS / Pinecone) and retrieve relevant context semantically using embeddings instead of scanning a limited set of messages.



