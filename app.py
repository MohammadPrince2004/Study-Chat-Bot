import torch
import streamlit as st
from pypdf import PdfReader
from transformers import pipeline
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- page setup
st.set_page_config(page_title="Study Chat Bot", page_icon="📖", layout="centered")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 4rem; max-width: 850px;}
.stButton > button {border-radius: 10px; padding: 0.5rem 1.2rem;}
</style>
""", unsafe_allow_html=True)

FORMAT = '''Return ONLY a JSON list, with no extra text. Each item must look like:
{"topic": "...", "question": "...", "options": ["...", "...", "...", "..."], "answer": "one of the options, copied exactly", "explanation": "one sentence"}'''


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model (first time takes a few minutes)...")
def get_llm():
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct",
                    torch_dtype=torch.float16, device_map="auto",
                    max_new_tokens=2000, do_sample=True, temperature=0.4,
                    return_full_text=False)
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))


# ---------------------------------------------------------------- PDF + summary
def read_pdf(file):
    return "\n".join(p.extract_text() or "" for p in PdfReader(file).pages)


def summarize(llm, text, language="English", max_parts=3):
    chain = ChatPromptTemplate.from_template(
        "Summarize the following text as clear bullet points grouped by topic. "
        "Use ONLY the information in the text. Write in {language}.\n\n{text}"
    ) | llm | StrOutputParser()
    parts = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=200).split_text(text)[:max_parts]
    partial = chain.batch([{"text": p, "language": language} for p in parts])   # Map
    if len(partial) == 1:
        return partial[0]
    return chain.invoke({"text": "\n".join(partial)[:7000], "language": language})   # Reduce


# ---------------------------------------------------------------- RAG chat
@st.cache_resource(show_spinner="Building search index...")
def build_store(text):
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(text)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return FAISS.from_texts(chunks, emb)


def ask(llm, store, question, k=4):
    docs = store.similarity_search(question, k=k)
    context = "\n\n".join(d.page_content for d in docs)
    chain = ChatPromptTemplate.from_template(
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say: I could not find this in the lecture.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    ) | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question}), docs


def show_message(m):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("📚 Sources"):
                for s in m["sources"]:
                    st.caption(s[:300])


# ---------------------------------------------------------------- quiz
def is_valid(q):
    keys = ("topic", "question", "options", "answer", "explanation")
    return (isinstance(q, dict)
            and all(k in q for k in keys)
            and isinstance(q["options"], list)
            and len(q["options"]) == 4
            and isinstance(q["answer"], str)
            and q["answer"].strip() in [str(o).strip() for o in q["options"]])


def generate_quiz(llm, summary, n=5, level="medium", language="English", retries=2):
    chain = PromptTemplate.from_template(
        "Create {n} {level} multiple-choice questions based ONLY on the text. "
        "Do not reveal the answer inside the question. Write in {language}.\n{format}\n\nText:\n{text}"
    ) | llm | JsonOutputParser()

    for _ in range(retries + 1):
        try:
            quiz = chain.invoke({"n": n, "level": level, "language": language,
                                 "format": FORMAT, "text": summary[:6000]})
            if isinstance(quiz, dict):
                quiz = quiz.get("questions", [])
            good = [q for q in quiz if is_valid(q)]
            if good:
                for q in good:
                    q["options"] = [str(o).strip() for o in q["options"]]
                    q["answer"] = q["answer"].strip()
                return good[:n]
        except Exception:
            pass
    return []


# ---------------------------------------------------------------- state + sidebar
ss = st.session_state
ss.setdefault("stage", "upload")
ss.setdefault("chat", [])

with st.sidebar:
    st.header("📖 Study Chat Bot")
    st.caption("Summarize a lecture, chat with it, then test yourself.")
    if ss.stage != "upload":
        if st.button("📄 New lecture", key="new_lecture"):
            for k in ["summary", "text", "questions", "answers", "qi"]:
                ss.pop(k, None)
            ss.chat = []
            ss.stage = "upload"
            st.rerun()


# ---------------------------------------------------------------- screens
if ss.stage == "upload":
    st.title("📖 Study Chat Bot")
    st.caption("Upload a lecture, get a summary, ask questions, then test yourself.")
    pdf = st.file_uploader("Upload your lecture (PDF)", type="pdf")
    if st.button("Summarize", type="primary", key="summarize_btn") and pdf:
        text = read_pdf(pdf)
        if len(text.strip()) < 200:
            st.warning("The PDF has almost no text (maybe scanned images).")
        else:
            llm = get_llm()
            with st.spinner("Summarizing..."):
                ss.summary = summarize(llm, text)
                ss.text = text
            ss.stage = "summary"
            st.rerun()

elif ss.stage == "summary":
    st.title("📖 Study Chat Bot")
    tab_sum, tab_chat, tab_quiz = st.tabs(["📝 Summary", "💬 Chat", "🧠 Quiz"])

    with tab_sum:
        st.markdown(ss.summary)

    with tab_chat:
        if not ss.chat:
            st.info("Ask anything about the lecture. Answers come only from your file.")
        for m in ss.chat:
            show_message(m)
        if question := st.chat_input("Ask about the lecture..."):
            show_message({"role": "user", "content": question})
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, docs = ask(get_llm(), build_store(ss.text), question)
            ss.chat += [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer,
                 "sources": [d.page_content for d in docs]},
            ]
            st.rerun()

    with tab_quiz:
        n = st.slider("Number of questions", 3, 10, 5)
        if st.button("Start quiz", type="primary", key="start_quiz_btn"):
            with st.spinner("Generating quiz..."):
                ss.questions = generate_quiz(get_llm(), ss.summary, n)
            if not ss.questions:
                st.error("Could not generate a valid quiz. Try again.")
            else:
                ss.stage = "quiz"
                ss.qi = 0
                ss.answers = []
                st.rerun()

elif ss.stage == "quiz":
    questions = ss.questions
    q = questions[ss.qi]
    st.progress(ss.qi / len(questions), text=f"Question {ss.qi + 1} of {len(questions)}")
    with st.container(border=True):
        st.subheader(q["question"])
        choice = st.radio("Choose one", q["options"], index=None,
                          key=f"r{ss.qi}", label_visibility="collapsed")
    if st.button("Confirm ✅", type="primary", key="confirm_btn"):
        if choice is None:
            st.warning("Choose an answer first")
        else:
            ss.answers.append(choice)
            ss.qi += 1
            if ss.qi == len(questions):
                ss.stage = "result"
            st.rerun()

elif ss.stage == "result":
    questions = ss.questions
    score = sum(a == q["answer"] for q, a in zip(questions, ss.answers))
    pct = score / len(questions)

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{score}/{len(questions)}")
    c2.metric("Percent", f"{pct:.0%}")
    c3.metric("Grade", "🏆 Excellent" if pct >= .9 else "👏 Very good" if pct >= .75
              else "👍 Good" if pct >= .6 else "📚 Review")
    st.progress(pct)
    if pct >= .8:
        st.balloons()

    for i, (q, a) in enumerate(zip(questions, ss.answers), 1):
        ok = a == q["answer"]
        with st.expander(f"{'✅' if ok else '❌'}  {i}. {q['question']}"):
            st.write(f"**Your answer:** {a}")
            if not ok:
                st.write(f"**Correct answer:** {q['answer']}")
            st.caption(q["explanation"])

    if st.button("⬅️ Back to lecture", key="back_btn"):
        ss.stage = "summary"
        st.rerun()
