from define_state import GraphState
from utils import rag_chain, retrieval_grader, question_rewriter, question_rewriter_to_search
from langchain.schema import Document
from define_model import web_search
from build_vectostores import retriever

def retrieve(state):
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.get_relevant_documents(question)
    return {"documents": documents,"question":question}

def generate(state):
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents,"question": question, "generation": generation}

def evaluate_documents(state):
    print("---CHECK DOCUMENT RELEVANT TO QUESTION")
    question = state["question"]
    documents = state["documents"]
    filter_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "documents": documents})
        grade = score.binary_score
        if grade == "Yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filter_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    if len(filter_docs)/len(documents) <= 0.7:
        web_search = "Yes"
    return{"documents": documents, "question": question, "web_search": web_search}

def transform_query(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def transform_query_to_search(state):
    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter_to_search.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    docs = web_search.invoke({"question": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content = web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENT---")
    web_search = state["web_search"]
    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE---")
        return "generate"



