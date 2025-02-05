# AI-Powered RAG Pipeline with Web Search and Grading

## Overview
This project implements an AI-powered retrieval-augmented generation (RAG) pipeline that intelligently routes user queries to a vector store or web search based on relevance. It utilizes various models and tools for document retrieval, response generation, and quality assessment.

## Features
- **Vector Store & Web Search**: Routes queries to either a FAISS vector store or a web search tool.
- **LangChain Integration**: Utilizes LangChain's document loaders, retrievers, and LLM integrations.
- **Document Splitting & Embeddings**: Uses RecursiveCharacterTextSplitter for chunking and OllamaEmbeddings for vector representation.
- **RAG-based Answer Generation**: Fetches documents and generates responses using Llama 3.2.
- **Quality Grading & Hallucination Detection**: Evaluates document relevance, hallucination levels, and correctness using an LLM.
- **StateGraph Workflow**: Implements a structured workflow to decide between retrieval, search, and response generation.

## Installation
### Prerequisites
- Python 3.8+
- `pip install` the required libraries:
  ```sh
  pip install langchain langchain-core langchain-community langchain-ollama faiss-cpu python-dotenv
  ```
- Set up environment variables:
  ```sh
  TAVILY_API_KEY=<your_api_key>
  ```

## Usage
### 1. Load Required Libraries
```python
import os
import json
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph
from IPython.display import Image, display
from dotenv import load_dotenv
```

### 2. Initialize Models and API Keys
```python
load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")
web_search_tools = TavilySearchResults(k=3)
llm = ChatOllama(model='llama3.2:3b')
llm_json = ChatOllama(model='llama3.2:3b', format='json', temperature=0)
```

### 3. Load and Process Documents
```python
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
)
docs_split = text_splitter.split_documents(docs_list)
```

### 4. Create a Vector Store and Retriever
```python
vectorstore = FAISS.from_documents(docs_split, embedding=OllamaEmbeddings(model="nomic-embed-text"))
retriever = vectorstore.as_retriever(k=3)
```

### 5. Define Query Routing Logic
```python
router_instructions = """
You are an expert at routing a user question to a vector store or web search.
Use 'vectorstore' for agents, prompt engineering, and adversarial attacks.
Use 'websearch' for all else.
Return a JSON with {'datasource': 'websearch' or 'vectorstore'}.
"""
```

### 6. Implement the RAG Workflow
```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents}

def generate(state):
    question = state["question"]
    documents = format_docs(state["documents"])
    rag_prompt = f"""You are an assistant...\nContext: {documents}\nQuestion: {question}\n"""
    generation = llm.invoke([HumanMessage(content=rag_prompt)])
    return {"generation": generation}
```

### 7. Define a StateGraph Workflow
```python
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.set_conditional_entry_point(route_question, {"websearch": "websearch", "vectorstore": "retrieve"})
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
```

### 8. Execute the Workflow
```python
inputs = {"question": "What are the types of agent memory?"}
for event in graph.stream(inputs, stream_mode="values"):
    print(event)
```

## Future Enhancements
- Expand the dataset with more diverse documents.
- Implement caching to reduce API call costs.
- Improve answer grading with reinforcement learning.

## License
This project is open-source under the MIT License.

