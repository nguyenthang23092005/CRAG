from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from define_model import llm_google
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel,Field

from define_model import llm_google

rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | llm_google | StrOutputParser


class RetrievalEvaluator(BaseModel):
    binary_score: str = Field(description = "Documents are relevant to the question,'Yes' or 'No'")
structured_llm_evaluator = llm_google.with_structured_output(RetrievalEvaluator)

system_evaluator = """You are a document retrieval evaluator that's responsible for checking the relevancy of a retrieved document to the user's question. \\n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \\n
    Output a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""
retrieval_evaluator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_evaluator),
        ("human", "Retrieved document: \n {document} \n User question: {question}")
    ]
)
retrieval_grader = retrieval_evaluator_prompt | structured_llm_evaluator


system_rewrite = """You are a question re-writer that converts an input question to a better version that is optimized for retrieve to relevant documents
"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_rewrite),
        ("human", "Here is the initial question: \n {question} \n Formulate an improved question.")
    ]
)
question_rewriter = re_write_prompt | llm_google | StrOutputParser


system_rewrite_to_search = """You are a question re-writer that converts an input question to a better version that is optimized 
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt_to_search = ChatPromptTemplate.from_messages(
    [
        ("system",system_rewrite_to_search),
        ("human", "Here is the initial question: \n {question} \n Formulate an improved question.")
    ]
)
question_rewriter_to_search = re_write_prompt_to_search | llm_google | StrOutputParser



