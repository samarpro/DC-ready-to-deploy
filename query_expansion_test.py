# Expansion of app1.py

import streamlit as st
from typing import List
from google.genai import Client
from qdrant_client import QdrantClient
from langchain_voyageai import VoyageAIEmbeddings
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client.models import models
from langchain_groq import ChatGroq
import os
import re

from dotenv import load_dotenv

load_dotenv()
global_meta_dict = {}
print(st.secrets["QDRANT_HOST"])
llm = Client(api_key=st.secrets["GOOGLE_API_KEY"])
qe_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4, max_retries=3)
vc = VoyageAIEmbeddings(model="voyage-3", api_key=st.secrets["VOYAGE_API_KEY"])
qclient = QdrantClient(
    url=st.secrets["QDRANT_HOST"],
    api_key=st.secrets["QDRANT_API_KEY"],
    https=True,
    timeout=100,
    prefer_grpc=True,
    check_compatibility=False,
)

print("Qdrant client has been configured.")
# configuring sparse and late embedding models
sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm25")
# late_iteraction_model = LateInteractionTextEmbedding("colbert-ir/colbertv2.0")

print("Sparse embedding is done as well.")


def convert_links_to_markdown(text, link_dict):

    def replace_link(match):
        link_text = match.group(1).strip()  # Extract and trim spaces
        normalized_text = " ".join(
            link_text.split()
        )  # Normalize spaces (removes extra spaces)
        url = link_dict.get(normalized_text, "#")  # Lookup in dictionary
        return f"[{normalized_text}]({url})"  # Convert to Markdown link

    # Regex to find <link> text </link> (handling spaces)
    pattern = r"<link>\s*(.*?)\s*</link>"
    return re.sub(pattern, replace_link, text)


# ---- Function: Simulate RAG retrieval (Replace with actual retrieval logic) ----
def retrieve_documents(query: str) -> List[str]:
    """Simulate retrieving relevant documents based on a query."""
    dense_query = vc.embed_query(query)
    sparse_query = next(sparse_embedding_model.query_embed(query))
    # late_query = next(late_iteraction_model.query_embed(query))

    prefetch = [
        models.Prefetch(query=dense_query, using="voyage-3", limit=50),
        models.Prefetch(
            query=models.SparseVector(**sparse_query.as_object()),
            using="bm25",
            limit=50,
        ),
    ]
    results = qclient.query_points(
        "hybrid-search",
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
        limit=10,
    )

    list_retrieved_docs = []

    for point in results.points:
        list_retrieved_docs.append(point.payload["document"])
        global_meta_dict.update(point.payload["metadata"])

    return list_retrieved_docs


# ---- Function: Simulate Response Generation (Replace with actual AI model) ----
def generate_response(query: str, retrieved_docs: List[str]) -> str:
    """Simulate AI-generated response using retrieved documents."""
    context = ""
    for idx, points in enumerate(retrieved_docs):
        print(f"-------------{idx}-------------------")
        context += points

    prompt = f"""
        System Message:

        You are a helpful assistant in a student-facing application that answers university-related queries using only the information provided in the context below.

Your goal is to deliver answers that are accurate, student-friendly, and well-structured. Respond in a tone that is clear, reassuring, and informative — like an experienced academic advisor.

 Answer Format Guidelines:
Start with a direct answer or summary, so students get immediate clarity.

Use numbered steps for processes or how-to questions.

Use bullet points for conditions, options, or eligibility criteria.

Use short paragraphs for background, implications, or rationale.

Include any definitions for terms that may be unfamiliar.
Content & Logic Guidelines:
Prioritize the most important or general-use information first.

Rephrase dense or policy-heavy content into plain, student-accessible language.

If the context is incomplete:

Explain what is known.

State clearly what is unknown.

Suggest contacting a relevant office or advisor for clarification.
 Constraints:
Use only the information in the context.

Do not fabricate or assume any missing information.

Do not omit or modify any <link></link> tags — include them exactly as provided.

Always end your answer with a helpful recommendation to consult the most relevant office or representative (e.g., academic advisor, registrar, IT helpdesk).
        Context:
        {context}

        User Query:
        {query}
    """

    resp = llm.models.generate_content(model="gemini-2.0-flash", contents=[prompt])
    return resp.text


# ---- Streamlit App ----
st.set_page_config(page_title="Deakin College Chatbot v12", layout="wide")

# ---- Sidebar for Referenced Documents ----
st.sidebar.title("📄 Referenced Documents")
st.sidebar.write("These are the documents retrieved by the RAG model:")

# ---- Main Chat Interface ----
st.title("🤖 Deakin College Chatbot v12")
st.write("Ask me anything about Deakin College!")

# Input field for user query
query = st.text_input("Type your question here:")


def perform_query_expansion(query):
    prompt = f"""
        You rewrite student queries to improve retrieval quality in a dense-embedding-based RAG system for an educational institution.

        Your goal is to convert informal, vague, or conversational inputs into clear, content-rich, standalone queries.

        Guidelines:
        Remove vague references like “this,” “it,” or “that” — always be explicit.
        Make the query self-contained and academically phrased.
        Clarify what the student is asking: definition, comparison, explanation, example, process, etc.
        Include any key terms or constraints that are implied.
        Avoid chatbot-style phrases (“Can you tell me…”).
        ➤ Output only the rewritten query.
        Query:
        {query}
        """
    msg = qe_llm.invoke(prompt)
    print("----- Query: ", msg)
    return msg.content


if query:
    # Retrieve relevant documents
    query = perform_query_expansion(query)
    retrieved_docs = retrieve_documents(query)

    # Display retrieved documents in the sidebar
    for doc in retrieved_docs:
        st.sidebar.markdown(f"- {doc}")

    # Generate response
    response = generate_response(query, retrieved_docs)

    final_resp = convert_links_to_markdown(response, global_meta_dict)

    # Display chatbot response
    st.markdown("### 🤖 Chatbot Response")
    st.write(final_resp)

    # Optional: Allow user to refine search
    if st.button("🔄 Refine Search"):
        query += "Explain briefly about the topic."
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*Deakin College Chatbot v12 - Powered by RAG* 🚀")
print("Everything ran.")
