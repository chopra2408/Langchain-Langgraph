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
from langgraph.graph import END
from langgraph.graph import StateGraph
from IPython.display import Image, display
from typing import List
from typing_extensions import TypedDict
from dotenv import load_dotenv

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")

web_search_tools = TavilySearchResults(k=3)
local_llm = 'llama3.2:3b'
llm = ChatOllama(model=local_llm)
llm_json = ChatOllama(model=local_llm, format='json', temperature=0)

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

vectorstore = FAISS.from_documents(
    docs_split, embedding=OllamaEmbeddings(model="nomic-embed-text")
)
retriever = vectorstore.as_retriever(k=3)

#result = retriever.invoke("agent memory")
#print(result)

router_instructions = """
You are an expert at routing a user question to a vector store or web search.
The vectorstore Contains documents related to agents, prompt engineering and adversarial attacks.
Use the vector store for questions on these topics. For all else, and especially for current events, use web search.
Return a json with single key, 'datasource' that is 'websearch' or 'vectorstore', depending on the question.
"""

question = [HumanMessage(content="What are the types of agent memory?")]
test_vector_store = llm_json.invoke(
    [SystemMessage(content=router_instructions)] + question
)
#print(json.loads(test_vector_store.content))

doc_grader_instructions = """
You are a grader assessing relevance af a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
"""

doc_grade_prompt = """
Here is the retrieved document: \n\n {document}
Here is the user question: \n\n{question}
This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
Return json with single key, binary_score that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant or not.
"""

question = "what is chain of thought prompting?"
docs = retriever.invoke(question)
doc_txt = docs[1].page_content
doc_grade_prompt_formatted = doc_grade_prompt.format(
    document=doc_txt, question=question
)
result = llm_json.invoke(
    [
        SystemMessage(content=doc_grader_instructions)
    ]
    + 
    [
        HumanMessage(content=doc_grade_prompt_formatted)
    ]
)
#print(json.loads(result.content))

rag_prompt = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
Think carefully about the above context.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

docs = retriever.invoke(question)
docs_txt = format_docs(docs)    
rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
#print(generation.content)

#print(web_search_tools.invoke("LLM agents"))

hallucinantion_grader_instructions = """ 
You are a teacher, grading a quiz.
You will be given FACTS and STUDENT ANSWERS.
Here is a great criteria to follow:

(1) Ensure the STUDENT ANSWERS is grounder in the FACTS.
(2) Ensure the STUDENT ANSWERS does not contain "hallucinated" information outside the scope of the FACTS.

Score:
A score of yes means that the students answer meets all the criteria. This is the highest (best) score.

A score of no means that students answer does not meet all the criteria. This is lowest possible score you can give.

Explain your reasoning in a step to step manner to ensure your reasoning and conclusion are correct.
Avoid simply stating the correct answer at the outset
"""

hallucinantion_grader_prompt = """
FACTS" \n\n {documents} \n\n STUDENT ANSWER: {generation}.
Return json with two keys, binary_score is 'yes' or 'no' score to indicate whether the student answer meets all the criteria.
And a key, 'datasource' that is 'websearch' or 'vectorstore' depending upopn the facts used.
"""

hallucinantion_grader_prompt_formatted = hallucinantion_grader_prompt.format(
    documents=docs_txt, generation=generation.content
)

result = llm_json.invoke(
    [
        SystemMessage(content=hallucinantion_grader_instructions)
    ]
    +
    [
        HumanMessage(content=hallucinantion_grader_prompt_formatted)
    ]
)   

#print(json.loads(result.content))

answer_grader_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:

(1) The STUDENT ANSWER helps to answer the QUESTION

Score:

A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

Avoid simply stating the correct answer at the outset."""

answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

question = "What are the vision models released today as part of Llama 3.2?"
answer = "The Llama 3.2 models released today include two vision models: Llama 3.2 11B Vision Instruct and Llama 3.2 90B Vision Instruct, which are available on Azure AI Model Catalog via managed compute. These models are part of Meta's first foray into multimodal AI and rival closed models like Anthropic's Claude 3 Haiku and OpenAI's GPT-4o mini in visual reasoning. They replace the older text-only Llama 3.1 models."

answer_grader_prompt_formatted = answer_grader_prompt.format(
    question=question, generation=answer
)
result = llm_json.invoke(
    [SystemMessage(content=answer_grader_instructions)]
    + [HumanMessage(content=answer_grader_prompt_formatted)]
)
#print(json.loads(result.content))

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str   
    web_search: str
    answers: int
    max_retries: int
    loop_step: int
    documents: List[str] 
    
def retrieve(state: GraphState):  
    """
    Retrieve documents from vectorstore
    """
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents}

def generate(state: GraphState):   
    """
    Generate answer using RAG
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    docs_txt = format_docs(documents)   
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation_content = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
   
    return {"generation": generation_content, "loop_step": loop_step + 1}
    
def grade_documents(state: GraphState):  
    """
    Determines whether the retrieved documents are relevant to the question
    """
    print("---CHECK DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = "No"
    for d in documents:
        doc_grade_prompt_formatted = doc_grade_prompt.format(
            document=d.page_content, question=question
        )
        result = llm_json.invoke([SystemMessage(content=doc_grader_instructions)] + [HumanMessage(content=doc_grade_prompt_formatted)])
        grade = json.loads(result.content)['binary_score']

        if grade.lower() == "yes":  
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "web_search": web_search}

def web_search(state: GraphState):  
    """
    Searches the web for the answer to the question
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    docs = web_search_tools.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)   
    documents.append(web_results)
    return {"documents": documents}


def route_question(state):
    """
    Route question to web search or RAG.
    """
    print("---ROUTER---")
    question = state["question"]
    router_question = llm_json.invoke(
        [SystemMessage(content=router_instructions)] + [HumanMessage(content=question)]
    )
    source = json.loads(router_question.content)['datasource']
    
    if source == "websearch":
        print("---ROUTER: WEB SEARCH---")
        return "websearch"
    
    elif source == "vectorstore":
        print("---ROUTER: VECTOR STORE---")
        return "vectorstore"
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer or re-try
    """
    print("---ASSESS GRADED ANSWER---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"] 
    
    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS NOT RELVANT, WEB SEARCH---")
        return "websearch"
    
    else:
        print("---DECISION: GENERATE---")
        return "generate"   
    
def grade_generation_v_documents_and_question(state: GraphState):  
    """
    Determines whether the generation is gated on the question.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]  
    max_retries = state.get("max_retries", 3)

    hallucinantion_grader_prompt_formatted = hallucinantion_grader_prompt.format(
        documents=format_docs(documents), generation=generation  
    )
    result = llm_json.invoke(
        [SystemMessage(content=hallucinantion_grader_instructions)]
        + [HumanMessage(content=hallucinantion_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)['binary_score']
    
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION")
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=generation
        )
        result = llm_json.invoke(
            [SystemMessage(content=answer_grader_instructions)]
            + [HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)['binary_score']

        if grade == "yes":
            print("---DECISION: GENERATION GROUNDED IN QUESTION---")
            return "useful"

        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOESNOT ADDRESS QUESTION---")
            return "not useful"
        
        else:
            print("--DECISION: MAX RETIES REACHED")
            return "max_retries"
        
    elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
        
    else:
        print("--DECISION: MAX RETIES REACHED")
        return "max_retries"
 

workflow = StateGraph(GraphState)

workflow.add_node("websearch", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_document", grade_documents)
workflow.add_node("generate", generate)

workflow.set_conditional_entry_point(
    route_question, 
    {
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)

workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_document")
workflow.add_conditional_edges(
    "grade_document",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "not useful": "websearch",
        "not supported": "websearch",
        "max_retries": END,
    },
)

graph = workflow.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

inputs = {"question": "What are the types of agent memory?", "max_retries": 3}
for event in graph.stream(inputs, stream_mode="values"):
    print(event)