import logging
from typing import TypedDict, List, Literal
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str
    retries: int
    verdict: Literal["accept", "retry"]
    citations: List[str]


def retriever_agent(state, vector_db):
    logger.info("Retriever agent running")
    docs = vector_db.similarity_search(state["question"], k=4)
    return {"retrieved_docs": docs}


def answer_agent(state):
    logger.info("Answer agent running")
    llm = ChatOpenAI(temperature=0)

    context = "\n\n".join(d.page_content for d in state["retrieved_docs"])
    prompt = f"""
Answer ONLY using context below.
If answer is not present, say "I don't know".

Context:
{context}

Question:
{state["question"]}
"""
    return {"answer": llm.predict(prompt)}


def critic_agent(state):
    logger.info("Critic agent running")

    if state["retries"] >= 1:
        logger.info("Max retries reached → accepting")
        return {"verdict": "accept"}

    llm = ChatOpenAI(temperature=0)
    context = "\n\n".join(d.page_content for d in state["retrieved_docs"])

    prompt = f"""
Is the answer supported by context?
Reply: accept or retry

Context:
{context}

Answer:
{state["answer"]}
"""
    verdict = llm.predict(prompt).strip().lower()
    return {
        "verdict": verdict,
        "retries": state["retries"] + 1
    }


def citation_agent(state):
    logger.info("Citation agent running")
    seen = set()
    citations = []

    for d in state["retrieved_docs"]:
        key = (d.metadata["source"], d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)

        if "page" in d.metadata:
            citations.append(f'{d.metadata["source"]} (Page {d.metadata["page"]})')
        else:
            citations.append(d.metadata["source"])

    return {"citations": citations}


def build_langgraph(vector_db):
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", lambda s: retriever_agent(s, vector_db))
    graph.add_node("answer", answer_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("cite", citation_agent)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "critic")

    graph.add_conditional_edges(
        "critic",
        lambda s: s["verdict"],
        {"accept": "cite", "retry": "answer"}
    )

    graph.add_edge("cite", END)
    return graph.compile()
