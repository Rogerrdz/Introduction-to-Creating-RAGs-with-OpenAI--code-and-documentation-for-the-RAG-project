# Introduction to Creating RAGs with OpenAI
## code-and-documentation for the RAG project

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, Pinecone as the vector database, and Google Gemini as the embedding model and LLM.

---

## Architecture
```
Web Document
     ↓
WebBaseLoader  →  loads raw text from a URL
     ↓
RecursiveCharacterTextSplitter  →  splits text into chunks
     ↓
GoogleGenerativeAIEmbeddings  →  converts chunks into vectors
     ↓
Pinecone  →  stores and indexes the vectors
     ↓
User Question
     ↓
Retriever  →  finds the most similar chunks in Pinecone
     ↓
ChatPromptTemplate  →  combines context + question
     ↓
ChatGoogleGenerativeAI (gemini-2.5-flash)  →  generates the answer
     ↓
Final Answer
```

### Components

| Component | Description |
|---|---|
| `WebBaseLoader` | Loads documents from a web URL |
| `RecursiveCharacterTextSplitter` | Splits documents into smaller chunks |
| `GoogleGenerativeAIEmbeddings` | Converts text into vector embeddings using Gemini |
| `Pinecone` | Vector database that stores and retrieves embeddings |
| `PineconeVectorStore` | LangChain integration for Pinecone |
| `ChatGoogleGenerativeAI` | Google Gemini LLM for generating answers |
| `ChatPromptTemplate` | Structures the prompt with context and question |
| `StrOutputParser` | Parses the model output into plain string |

---

## Project Structure
```
repo-2-rag-pinecone/
│
├── rag_pinecone.ipynb   <- Main Jupyter notebook with all the code
├── .env.example         <- Template for environment variables (no real keys)
├── requirements.txt     <- Python dependencies
└── README.md            <- Project documentation
```

---

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/Rogerrdz/Introduction-to-Creating-RAGs-with-OpenAI--code-and-documentation-for-the-RAG-project.git
cd Introduction-to-Creating-RAGs-with-OpenAI--code-and-documentation-for-the-RAG-project.git
```

### 2. Install dependencies
```bash
pip install langchain langchain-google-genai langchain-pinecone pinecone langchain-community bs4 python-dotenv
```

Or using the requirements file:
```bash
pip install -r requirements.txt
```

### 3. Set up your API keys

Copy the `.env.example` file and rename it to `.env`:
```bash
cp .env.example .env
```

Then fill in your API keys inside the `.env` file:
```
GOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

Where to get the keys:
- Google API Key -> aistudio.google.com -> Get API Key (free)
- Pinecone API Key -> pinecone.io -> API Keys (free tier available)

### 4. Run the notebook
```bash
jupyter notebook rag_pinecone.ipynb
```

Run all cells in order from top to bottom.

---

## What the Notebook Covers

### Cell 1 - Install Dependencies
Installs all required libraries using pip.

### Cell 2 - Load Environment Variables
Safely loads API keys from the `.env` file using `python-dotenv`.

### Cell 3 - Load Document from the Web
Uses `WebBaseLoader` to load a web article about AI agents.
```python
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",)
)
docs = loader.load()
```

### Cell 4 - Split into Chunks
Uses `RecursiveCharacterTextSplitter` to divide the document into smaller pieces.
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)
```

### Cell 5 - Create Embeddings
Converts text chunks into vectors using Google Gemini embeddings.
```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```

### Cell 6 - Store in Pinecone
Creates a Pinecone index and uploads all document vectors.
```python
vector_store = PineconeVectorStore(embedding=embeddings, index=index)
vector_store.add_documents(splits)
```

### Cell 7 - Create the Retriever
Creates a retriever that finds the 3 most similar chunks for any given question.
```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

### Cell 8 - Initialize the LLM
Creates a `ChatGoogleGenerativeAI` instance using `gemini-2.5-flash`.

### Cell 9 - Build the RAG Chain
Combines the retriever, prompt, LLM, and output parser into a full chain.
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Cell 10 - Ask Questions
Sends questions to the RAG chain and prints the answers.
```python
answer = rag_chain.invoke("What is an AI agent?")
print(answer)
```

---

## Example Output
```
Q: What is an AI agent?
A: An AI agent is an autonomous system that perceives its environment,
makes decisions, and takes actions to achieve specific goals. It typically
consists of a memory component, planning capabilities, and the ability
to use external tools.

Q: What are the main components of an AI agent?
A: The main components of an AI agent are memory (short-term and long-term),
planning (task decomposition and reflection), and action (tool use and
external API calls).
```

---

## requirements.txt
```
langchain
langchain-community
langchain-google-genai
langchain-pinecone
pinecone
bs4
python-dotenv
```

---

## Security Note

Never upload your `.env` file to GitHub. The `.gitignore` file in this project already excludes it. Only the `.env.example` file with empty values is safe to share publicly.

---

## References

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [LangChain Pinecone Integration](https://python.langchain.com/docs/integrations/vectorstores/pinecone)
- [Google AI Studio](https://aistudio.google.com)
- [Pinecone](https://pinecone.io)