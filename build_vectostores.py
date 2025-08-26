from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
import json
import uuid
import os


class BGEM3Embedding(Embeddings):
    def __init__(self, model_name="BAAI/bge-m3", use_fp16=True):
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def embed_documents(self, texts):
        embeddings = [self.model.encode(text)["dense_vecs"] for text in texts]
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, text):
        embedding = self.model.encode(text)["dense_vecs"]
        return embedding.tolist()
    

urls =[
    "https://ryanocm.substack.com/p/mystery-gift-box-049-law-1-fill-your",
    "https://ryanocm.substack.com/p/105-the-bagel-method-in-relationships",
    "https://ryanocm.substack.com/p/098-i-have-read-100-productivity",
]
persist_directory = "rag-schoma"

embeddings = BGEM3Embedding()

if os.path.exists(persist_directory):
    vectostore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="rag-schoma"
    )
else:
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectostore = Chroma.from_documents(
        documents=doc_splits,
        embedding=embeddings,
        collection_name="rag-schoma",
        persist_directory=persist_directory
    )
    vectostore.persist() 

retriever = vectostore.as_retriever()

def add_new_doc(text, metadata=None, similarity_threshold=0.9):
    new_doc = Document(page_content=text, metadata=metadata or {})
    candidates = retriever.get_relevant_documents(text)

    if candidates:
        query_emb = embeddings.embed_query(text)
        for doc in candidates:
            doc_emb = embeddings.embed_query(doc.page_content) 
            cos_sim = sum(a*b for a, b in zip(query_emb, doc_emb)) / (
                (sum(a*a for a in query_emb) ** 0.5) * (sum(b*b for b in doc_emb) ** 0.5)
            )
            if cos_sim >= similarity_threshold:
                print("---DOCUMENT WERE IN VECTOSTORE---\n", "Cosine similarity: ",round(cos_sim, 3),"\n")
                return False

    vectostore.add_documents([new_doc])
    print("---DOCUMENT ADDED IN VECTOSTORE---")


def add_conversation(question, answer, metadata=None):
    conv_obj = {
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": answer
    }
    conv_json = json.dumps(conv_obj, ensure_ascii=False, indent=2)
    new_doc = Document(
        page_content=conv_json,
        metadata=metadata or {"type": "conversation"}
    )
    vectostore.add_documents([new_doc])
    print("---CONVERSATION ADDED IN VECTOSTORE---\n")