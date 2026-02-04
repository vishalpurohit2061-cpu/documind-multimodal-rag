import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st

from rag_engine import (
    load_pdf,
    load_website,
    load_image,
    split_documents,
    create_vector_db
)
from langgraph_rag import build_langgraph


# ---------------- Page Config ----------------
st.set_page_config(
    page_title="DocuMind",
    layout="centered"
)

st.title("🧠 DocuMind – Multimodal RAG")

# ---------------- Session State ----------------
if "graph" not in st.session_state:
    st.session_state.graph = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "chat" not in st.session_state:
    st.session_state.chat = []


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📂 Add Knowledge Sources")

    pdfs = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    images = st.file_uploader(
        "Upload Images (JPG / PNG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    website_url = st.text_input("Website URL")

    build_btn = st.button("🔨 Build Knowledge Base")

    if build_btn:
        with st.status("Processing sources...", expanded=True) as status:
            docs = []

            # PDFs
            if pdfs:
                st.write("📄 Reading PDFs...")
                for pdf in pdfs:
                    docs.extend(load_pdf(pdf))

            # Images
            if images:
                st.write("🖼️ Reading Images...")
                for img in images:
                    docs.extend(load_image(img))

            # Website
            if website_url:
                st.write("🌐 Reading Website...")
                docs.extend(load_website(website_url))

            if not docs:
                st.warning("No valid content found.")
            else:
                st.write(f"✅ Documents collected: {len(docs)}")

                st.session_state.chunks = split_documents(docs)
                st.write(f"✂️ Chunks created: {len(st.session_state.chunks)}")

                vector_db = create_vector_db(st.session_state.chunks)
                st.session_state.graph = build_langgraph(vector_db)

                status.update(
                    label="Knowledge base ready ✅",
                    state="complete"
                )


# ---------------- Debug Info ----------------
with st.expander("🛠 Debug Info"):
    st.write("Graph ready:", st.session_state.graph is not None)
    if st.session_state.chunks:
        st.write("Chunks:", len(st.session_state.chunks))


# ---------------- Chat History ----------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg:
            with st.expander("📌 Sources"):
                for s in msg["sources"]:
                    st.write(s)


# ---------------- Chat Input ----------------
question = st.chat_input("Ask from PDFs, websites, or images...")

if question:
    if st.session_state.graph is None:
        st.warning("Please add sources and build the knowledge base first.")
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("🤖 Thinking..."):
            result = st.session_state.graph.invoke({
                "question": question,
                "retries": 0
            })

        answer = result.get("answer", "No answer.")
        citations = result.get("citations", [])

        with st.chat_message("assistant"):
            st.write(answer)
            if citations:
                with st.expander("📌 Sources"):
                    for c in citations:
                        st.write(c)

        st.session_state.chat.append({
            "role": "user",
            "content": question
        })
        st.session_state.chat.append({
            "role": "assistant",
            "content": answer,
            "sources": citations
        })
