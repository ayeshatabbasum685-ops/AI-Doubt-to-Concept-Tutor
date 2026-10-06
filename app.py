
import os
import re
import json
import time

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Doubt-to-Concept Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CLASSIC UI / ANIMATIONS
# ============================================================

st.markdown(
    """
    <style>
    /* ---------- Classic background ---------- */
    /* =====================================================
       MAIN CLASSIC DARK THEME
       ===================================================== */

    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(78, 115, 223, .18), transparent 30%),
            radial-gradient(circle at 85% 25%, rgba(145, 91, 205, .14), transparent 30%),
            linear-gradient(135deg, #0b1020 0%, #121a2f 48%, #0e1528 100%);
        color: #e8ecf7;
    }

    /* Main Streamlit text */
    .stApp p,
    .stApp span,
    .stApp label,
    .stApp div,
    .stApp li {
        color: #e8ecf7;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: #f7f9ff !important;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, .86);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090d1a 0%, #121a2d 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #edf1fb !important;
    }

    /* Text inputs / text areas */
    .stTextInput input,
    .stTextArea textarea {
        background: #1a2238 !important;
        color: #f4f7ff !important;
        border: 1px solid #35415f !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #9ba6bf !important;
        opacity: 1 !important;
    }

    /* Select boxes */
    [data-baseweb="select"] > div {
        background: #1a2238 !important;
        color: #f4f7ff !important;
        border-color: #35415f !important;
    }

    [data-baseweb="select"] * {
        color: #f4f7ff !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #171f34;
        border: 1px dashed #53617f;
        border-radius: 16px;
        padding: 8px;
    }

    [data-testid="stFileUploader"] * {
        color: #e8ecf7 !important;
    }

    /* Radio buttons */
    [data-testid="stRadio"] label {
        background: #171f34;
        border: 1px solid #303b57;
        border-radius: 12px;
        padding: 8px 12px;
        margin-bottom: 5px;
    }

    [data-testid="stRadio"] label:hover {
        background: #202a45;
    }

    /* Alerts */
    [data-testid="stAlert"] * {
        color: #172033 !important;
    }

    /* ---------- Hero ---------- */
    .hero {
        padding: 30px 34px;
        border-radius: 24px;
        margin-bottom: 25px;
        background: linear-gradient(135deg, #1c2340, #30395f);
        color: white;
        box-shadow: 0 14px 40px rgba(31, 38, 68, .18);
        animation: fadeUp .55s ease-out;
    }

    .hero h1 {
        margin: 0;
        font-size: 40px;
        font-weight: 800;
    }

    .hero p {
        margin: 9px 0 0;
        color: #dbe1f5;
        font-size: 17px;
    }

    /* ---------- Cards ---------- */
    .classic-card {
        background: linear-gradient(145deg, #171f34, #121a2d);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 20px;
        padding: 23px;
        box-shadow: 0 12px 32px rgba(0,0,0,.24);
        margin-bottom: 18px;
        animation: fadeUp .45s ease-out;
    }

    .classic-card p,
    .classic-card li,
    .classic-card span {
        color: #dce3f2;
    }

    .feature-card {
        min-height: 155px;
        background: linear-gradient(145deg, #182139, #121a2e);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 10px 28px rgba(0,0,0,.22);
        transition: .25s ease;
    }

    .feature-card p {
        color: #c9d2e5;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 30px rgba(31,41,55,.11);
    }

    .stat-card {
        background: linear-gradient(145deg, #182139, #121a2e);
        border-radius: 18px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,.08);
        box-shadow: 0 10px 28px rgba(0,0,0,.20);
        text-align: center;
    }

    .stat-number {
        font-size: 30px;
        font-weight: 800;
        color: #30395f;
    }

    .stat-label {
        color: #aeb8ce;
        font-size: 13px;
    }

    [data-testid="stMetricLabel"] {
        color: #aeb8ce !important;
    }

    [data-testid="stMetricValue"] {
        color: #f4f7ff !important;
    }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #aeb8ce !important;
    }

    /* ---------- Buttons: pop when touched ---------- */
    [data-testid="stButton"] button {
        border-radius: 13px !important;
        border: 1px solid #d8dceb !important;
        font-weight: 700 !important;
        transition: transform .12s ease, box-shadow .12s ease, background .2s ease !important;
        box-shadow: 0 4px 12px rgba(31,41,55,.07) !important;
    }

    [data-testid="stButton"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 9px 20px rgba(31,41,55,.12) !important;
    }

    [data-testid="stButton"] button:active {
        transform: scale(.94) !important;
        box-shadow: 0 2px 6px rgba(31,41,55,.10) !important;
    }

    /* ---------- Inputs ---------- */
    textarea, input {
        border-radius: 12px !important;
    }

    /* ---------- Quiz ---------- */
    .question-card {
        background: #171f34;
        border: 1px solid #303b57;
        border-radius: 18px;
        padding: 20px;
        margin: 14px 0;
        box-shadow: 0 7px 20px rgba(31,41,55,.05);
        animation: fadeUp .35s ease-out;
    }

    .question-number {
        color: #59648a;
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .8px;
    }

    /* ---------- Animation ---------- */
    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #737b8f;
        padding: 20px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

if not API_KEY:
    st.error(
        "GEMINI_API_KEY is missing. Create a .env file beside app.py "
        "and add: GEMINI_API_KEY=your_key"
    )
    st.stop()

client = genai.Client(api_key=API_KEY)

GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "page": "Home",
    "chat_history": [],
    "pdf_loaded": False,
    "pdf_name": "",
    "chunks_count": 0,
    "quiz_data": None,
    "quiz_topic": "",
    "quiz_submitted": False,
    "user_answers": {},
    "quiz_results": [],
    "total_questions_answered": 0,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# MODELS / DATABASE
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def create_chroma_collection():
    db = chromadb.PersistentClient(path="./chroma_db")
    return db.get_or_create_collection(name="study_material")


embedding_model = load_embedding_model()
collection = create_chroma_collection()


# ============================================================
# PDF FUNCTIONS
# ============================================================

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


def split_text(text, chunk_size=800, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def clear_pdf_database():
    try:
        existing = collection.get()
        ids = existing.get("ids", [])

        if ids:
            collection.delete(ids=ids)

    except Exception:
        pass


def add_pdf_to_database(text):
    chunks = split_text(text)

    if not chunks:
        return 0

    clear_pdf_database()

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    ).tolist()

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    return len(chunks)


def search_pdf(question, number_of_results=4):
    if not st.session_state.pdf_loaded:
        return ""

    try:
        total_chunks = collection.count()

        if total_chunks == 0:
            return ""

        n_results = min(number_of_results, total_chunks)

        query_embedding = embedding_model.encode(
            [question],
            show_progress_bar=False
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        documents = results.get("documents", [])

        if not documents:
            return ""

        flattened = []

        for group in documents:
            if isinstance(group, list):
                flattened.extend(group)
            elif group:
                flattened.append(group)

        return "\n\n".join(flattened)

    except Exception:
        return ""


# ============================================================
# GEMINI FUNCTIONS
# ============================================================

def is_temporary_model_error(error):
    message = str(error).upper()

    temporary_codes = (
        "503",
        "UNAVAILABLE",
        "429",
        "RESOURCE_EXHAUSTED",
        "500",
        "INTERNAL",
        "TIMEOUT",
    )

    return any(code in message for code in temporary_codes)


def generate_with_gemini(prompt):
    last_error = None

    for model_name in GEMINI_MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                answer = getattr(response, "text", None)

                if answer and answer.strip():
                    return answer.strip()

                raise RuntimeError(
                    f"Gemini returned an empty response using {model_name}."
                )

            except Exception as error:
                last_error = error

                if not is_temporary_model_error(error):
                    raise

                if attempt == 0:
                    time.sleep(2)

    raise RuntimeError(
        "Gemini is temporarily unavailable. Please try again."
    ) from last_error


def ask_gemini(doubt, subject, level, style, pdf_context):
    if pdf_context:
        source_instruction = f"""
Use the following study material when relevant.

STUDY MATERIAL:
{pdf_context}

Do not invent information that contradicts the study material.
If the study material does not contain the answer, use general
academic knowledge and clearly explain the concept.
"""
    else:
        source_instruction = """
No PDF study material is currently available.
Use general academic knowledge.
"""

    prompt = f"""
You are an intelligent AI Doubt-to-Concept Tutor.

Your purpose is not simply to answer a student's question.
Identify the underlying concept the student is struggling to understand.

Student level: {level}
Subject: {subject}
Preferred explanation style: {style}

Student doubt:
{doubt}

{source_instruction}

Give the response using exactly these sections:

## 🎯 Main Concept
## 🔎 What You Are Confused About
## 📚 Concept Explanation
## 💡 Easy Example
## 🌍 Real-World Connection
## ⚠️ Common Mistake
## 🧠 Quick Memory Trick
## 📝 Quick Check
## 🚀 What To Learn Next

Use simple, student-friendly language.
"""

    return generate_with_gemini(prompt)


def generate_quiz(concept, subject):
    prompt = f"""
You are an educational quiz generator.

Subject: {subject}
Concept: {concept}

Create exactly 5 multiple-choice questions.

Return ONLY valid JSON.
No Markdown.
No ```json.
No text outside the JSON.

Format:
[
  {{
    "question": "Question text",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "answer": "A",
    "explanation": "Short explanation."
  }}
]

Rules:
1. Exactly 5 questions.
2. Exactly 4 options per question.
3. Only one correct answer.
4. Answer must be A, B, C, or D.
5. Suitable for a college student.
6. Test understanding, not only memorization.
7. Keep explanations short and clear.
"""

    raw = generate_with_gemini(prompt)

    raw = re.sub(
        r"```json\s*|```\s*",
        "",
        raw,
        flags=re.IGNORECASE
    ).strip()

    start = raw.find("[")
    end = raw.rfind("]")

    if start == -1 or end == -1:
        raise ValueError("Gemini did not return valid quiz JSON.")

    quiz = json.loads(raw[start:end + 1])

    if not isinstance(quiz, list) or len(quiz) != 5:
        raise ValueError("Quiz must contain exactly 5 questions.")

    for q in quiz:
        if not all(
            key in q for key in
            ["question", "options", "answer", "explanation"]
        ):
            raise ValueError("Quiz question has missing fields.")

        for letter in ["A", "B", "C", "D"]:
            if letter not in q["options"]:
                raise ValueError("One or more quiz options are missing.")

        if q["answer"] not in ["A", "B", "C", "D"]:
            raise ValueError("Invalid quiz answer.")

    return quiz


# ============================================================
# NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown("## 🧠 Tutor")
    st.caption("AI Doubt-to-Concept Learning Studio")
    st.divider()

    pages = [
        "🏠 Home",
        "💬 Doubt Tutor",
        "📄 PDF Study",
        "📝 Quiz",
        "📚 History",
        "📈 Progress",
        "ℹ️ About",
    ]

    current_label = {
        "Home": "🏠 Home",
        "Doubt Tutor": "💬 Doubt Tutor",
        "PDF Study": "📄 PDF Study",
        "Quiz": "📝 Quiz",
        "History": "📚 History",
        "Progress": "📈 Progress",
        "About": "ℹ️ About",
    }.get(st.session_state.page, "🏠 Home")

    selected = st.radio(
        "Navigation",
        pages,
        index=pages.index(current_label),
        label_visibility="collapsed"
    )

    page_map = {
        "🏠 Home": "Home",
        "💬 Doubt Tutor": "Doubt Tutor",
        "📄 PDF Study": "PDF Study",
        "📝 Quiz": "Quiz",
        "📚 History": "History",
        "📈 Progress": "Progress",
        "ℹ️ About": "About",
    }

    if page_map[selected] != st.session_state.page:
        st.session_state.page = page_map[selected]
        st.rerun()

    st.divider()

    st.markdown("### ⚙️ Learning Settings")

    subject = st.selectbox(
        "📚 Subject",
        [
            "General",
            "Python",
            "C Programming",
            "Data Structures",
            "DBMS",
            "Computer Networks",
            "Artificial Intelligence",
            "Machine Learning",
            "Mathematics",
            "Physics",
            "Chemistry",
        ]
    )

    level = st.selectbox(
        "🎓 Student Level",
        ["Beginner", "Intermediate", "Advanced"]
    )

    style = st.selectbox(
        "✍️ Explanation Style",
        [
            "Very Simple",
            "Exam Friendly",
            "Detailed",
            "With Examples",
        ]
    )

    st.divider()

    if st.session_state.pdf_loaded:
        st.success("📄 PDF knowledge active")
        st.caption(st.session_state.pdf_name)
    else:
        st.info("📄 No PDF loaded")


# ============================================================
# HEADER
# ============================================================

def show_hero(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    show_hero(
        "🧠 AI Doubt-to-Concept Tutor",
        "Understand the concept behind your doubt — not just the answer."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.chat_history)}</div>
                <div class="stat-label">Doubts Solved</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{len(st.session_state.quiz_results)}</div>
                <div class="stat-label">Quizzes Completed</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        pdf_status = "Ready" if st.session_state.pdf_loaded else "None"
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{pdf_status}</div>
                <div class="stat-label">Study Material</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        if st.session_state.quiz_results:
            avg = sum(
                x["score"] / x["total"] * 100
                for x in st.session_state.quiz_results
            ) / len(st.session_state.quiz_results)
            avg_text = f"{avg:.0f}%"
        else:
            avg_text = "—"

        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-number">{avg_text}</div>
                <div class="stat-label">Quiz Average</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## ✨ What can you do?")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>💬 Understand Any Doubt</h3>
                <p>Ask a question and AI identifies the concept,
                explains it simply, gives examples, mistakes and a
                quick memory trick.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📄 Learn From Your PDF</h3>
                <p>Upload notes or study material. The tutor searches
                your PDF before answering your doubt.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    c3, c4 = st.columns(2)

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📝 Test Yourself</h3>
                <p>Generate a 5-question quiz, answer every question,
                then see your score and explanations.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📈 Track Progress</h3>
                <p>See your quiz performance and learning activity
                in one simple dashboard.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## 🚀 Start Learning")

    a, b, c = st.columns(3)

    with a:
        if st.button("💬 Ask a Doubt", use_container_width=True):
            st.session_state.page = "Doubt Tutor"
            st.toast("Opening Doubt Tutor ✨")
            st.rerun()

    with b:
        if st.button("📄 Upload Study PDF", use_container_width=True):
            st.session_state.page = "PDF Study"
            st.toast("Opening PDF Study 📚")
            st.rerun()

    with c:
        if st.button("📝 Take a Quiz", use_container_width=True):
            st.session_state.page = "Quiz"
            st.toast("Opening Quiz 🎯")
            st.rerun()


# ============================================================
# DOUBT TUTOR PAGE
# ============================================================

elif st.session_state.page == "Doubt Tutor":

    show_hero(
        "💬 Doubt Tutor",
        "Ask your doubt and discover the concept behind it."
    )

    st.markdown(
        '<div class="classic-card">',
        unsafe_allow_html=True
    )

    doubt = st.text_area(
        "What are you confused about?",
        placeholder=(
            "Example: I know what a Python loop is, "
            "but I don't understand when I should use a for loop "
            "instead of a while loop."
        ),
        height=170
    )

    use_pdf = st.checkbox(
        "📄 Use my uploaded PDF as study material",
        value=st.session_state.pdf_loaded,
        disabled=not st.session_state.pdf_loaded
    )

    if st.button(
        "🧠 Understand My Doubt",
        use_container_width=True
    ):
        if not doubt.strip():
            st.warning("Please enter your doubt first.")
        else:
            with st.spinner("🔍 Finding the concept behind your doubt..."):
                try:
                    context = search_pdf(doubt) if use_pdf else ""

                    answer = ask_gemini(
                        doubt,
                        subject,
                        level,
                        style,
                        context
                    )

                    st.session_state.chat_history.append(
                        {
                            "doubt": doubt,
                            "answer": answer
                        }
                    )

                    st.toast("Concept understood successfully! 🎉")

                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown(
                        '<div class="classic-card">',
                        unsafe_allow_html=True
                    )

                    st.markdown("## 🧠 AI Concept Analysis")
                    st.markdown(answer)

                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception as error:
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.error(f"❌ Error: {error}")
    else:
        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PDF STUDY PAGE
# ============================================================

elif st.session_state.page == "PDF Study":

    show_hero(
        "📄 PDF Study",
        "Turn your notes, textbooks and study material into an AI tutor."
    )

    st.markdown(
        '<div class="classic-card">',
        unsafe_allow_html=True
    )

    uploaded_pdf = st.file_uploader(
        "Upload your study material",
        type=["pdf"],
        help="Best for text-based PDFs."
    )

    if uploaded_pdf:
        if st.session_state.pdf_name != uploaded_pdf.name:

            with st.spinner("📚 Reading and indexing your PDF..."):
                try:
                    pdf_text = extract_pdf_text(uploaded_pdf)

                    if not pdf_text.strip():
                        st.error(
                            "No readable text was found. "
                            "Scanned PDFs may require OCR."
                        )
                        st.session_state.pdf_loaded = False
                    else:
                        count = add_pdf_to_database(pdf_text)

                        st.session_state.pdf_loaded = count > 0
                        st.session_state.pdf_name = uploaded_pdf.name
                        st.session_state.chunks_count = count

                        if count:
                            st.toast("PDF indexed successfully! 📚")
                            st.success(
                                f"✅ {uploaded_pdf.name} is ready. "
                                f"{count} knowledge sections indexed."
                            )
                        else:
                            st.error("No readable text was found.")

                except Exception as error:
                    st.session_state.pdf_loaded = False
                    st.error(f"❌ PDF processing error: {error}")

    if st.session_state.pdf_loaded:

        st.success(
            f"📄 Active: {st.session_state.pdf_name}"
        )

        st.write(
            f"**Knowledge sections:** "
            f"{st.session_state.chunks_count}"
        )

        if st.button(
            "🗑️ Remove Current PDF",
            use_container_width=True
        ):
            clear_pdf_database()
            st.session_state.pdf_loaded = False
            st.session_state.pdf_name = ""
            st.session_state.chunks_count = 0
            st.toast("PDF removed.")
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "💡 After uploading, go to **Doubt Tutor** and enable "
        "'Use my uploaded PDF as study material'."
    )


# ============================================================
# QUIZ PAGE
# ============================================================

elif st.session_state.page == "Quiz":

    show_hero(
        "📝 Smart Quiz",
        "Test your understanding and reveal answers only after submission."
    )

    st.markdown(
        '<div class="classic-card">',
        unsafe_allow_html=True
    )

    concept = st.text_input(
        "Enter a concept for your 5-question quiz",
        value=st.session_state.quiz_topic,
        placeholder="Example: Python loops"
    )

    if st.button(
        "🎯 Generate 5-Question Quiz",
        use_container_width=True
    ):

        if not concept.strip():
            st.warning("Please enter a concept first.")
        else:
            with st.spinner("🧠 Creating your quiz..."):
                try:
                    quiz = generate_quiz(
                        concept,
                        subject
                    )

                    st.session_state.quiz_data = quiz
                    st.session_state.quiz_topic = concept
                    st.session_state.quiz_submitted = False
                    st.session_state.user_answers = {}

                    st.toast("Quiz generated! Good luck 🎯")
                    st.rerun()

                except Exception as error:
                    st.error(f"❌ Quiz error: {error}")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.quiz_data:

        quiz = st.session_state.quiz_data

        if not st.session_state.quiz_submitted:

            st.info(
                "Answer all 5 questions. "
                "Correct answers stay hidden until you submit."
            )

            for i, question in enumerate(quiz):

                st.markdown(
                    f"""
                    <div class="question-card">
                        <div class="question-number">
                            Question {i + 1} of 5
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"### {question['question']}"
                )

                options = question["options"]

                selected = st.radio(
                    "Select one:",
                    [
                        f"A. {options['A']}",
                        f"B. {options['B']}",
                        f"C. {options['C']}",
                        f"D. {options['D']}",
                    ],
                    index=None,
                    key=f"quiz_option_{i}"
                )

                if selected:
                    st.session_state.user_answers[i] = selected[0]

            st.markdown("")

            if st.button(
                "✅ Submit Quiz",
                use_container_width=True
            ):

                if len(st.session_state.user_answers) < 5:
                    st.warning(
                        f"Please answer all 5 questions. "
                        f"You answered "
                        f"{len(st.session_state.user_answers)}/5."
                    )
                else:
                    st.session_state.quiz_submitted = True
                    st.rerun()

        else:

            score = 0

            for i, question in enumerate(quiz):
                if (
                    st.session_state.user_answers.get(i)
                    == question["answer"]
                ):
                    score += 1

            percentage = score / 5 * 100

            st.markdown("## 🏆 Quiz Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Score", f"{score}/5")

            with col2:
                st.metric("Percentage", f"{percentage:.0f}%")

            with col3:
                if score == 5:
                    grade = "Excellent"
                elif score >= 4:
                    grade = "Very Good"
                elif score >= 3:
                    grade = "Good"
                else:
                    grade = "Keep Practicing"

                st.metric("Performance", grade)

            st.progress(score / 5)

            if score == 5:
                st.success("🎉 Perfect! Excellent understanding!")
            elif score == 4:
                st.success("🌟 Great work! You understand this topic well.")
            elif score == 3:
                st.info("👏 Good attempt. Review the incorrect answers.")
            else:
                st.warning(
                    "💪 Keep practicing. Read the explanations and try again."
                )

            # Save result only once for this quiz
            result_id = f"{st.session_state.quiz_topic}_{id(quiz)}"

            if not any(
                r["id"] == result_id
                for r in st.session_state.quiz_results
            ):
                st.session_state.quiz_results.append(
                    {
                        "id": result_id,
                        "topic": st.session_state.quiz_topic,
                        "score": score,
                        "total": 5,
                    }
                )

            st.markdown("---")
            st.markdown("## 📚 Answer Review")

            for i, question in enumerate(quiz):

                user_answer = st.session_state.user_answers[i]
                correct_answer = question["answer"]
                options = question["options"]

                st.markdown(
                    f"### Question {i + 1}"
                )

                st.write(question["question"])

                st.write(
                    f"**Your answer:** "
                    f"{user_answer}. {options[user_answer]}"
                )

                st.write(
                    f"**Correct answer:** "
                    f"{correct_answer}. "
                    f"{options[correct_answer]}"
                )

                if user_answer == correct_answer:
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Incorrect")

                st.info(
                    f"💡 **Why:** {question['explanation']}"
                )

            st.markdown("---")

            c1, c2 = st.columns(2)

            with c1:
                if st.button(
                    "🔄 Take Another Quiz",
                    use_container_width=True
                ):
                    st.session_state.quiz_data = None
                    st.session_state.quiz_submitted = False
                    st.session_state.user_answers = {}
                    st.session_state.quiz_topic = ""
                    st.rerun()

            with c2:
                if st.button(
                    "🏠 Back Home",
                    use_container_width=True
                ):
                    st.session_state.page = "Home"
                    st.rerun()


# ============================================================
# HISTORY PAGE
# ============================================================

elif st.session_state.page == "History":

    show_hero(
        "📚 Doubt History",
        "Review the concepts you have already explored."
    )

    if not st.session_state.chat_history:

        st.info(
            "No doubts yet. Go to Doubt Tutor and ask your first question."
        )

    else:

        st.write(
            f"**{len(st.session_state.chat_history)} doubts solved**"
        )

        for i, item in enumerate(
            reversed(st.session_state.chat_history),
            1
        ):

            with st.expander(
                f"💬 Doubt {i}: {item['doubt']}"
            ):

                st.markdown(item["answer"])

        st.markdown("---")

        if st.button(
            "🗑️ Clear Doubt History",
            use_container_width=True
        ):
            st.session_state.chat_history = []
            st.toast("History cleared.")
            st.rerun()


# ============================================================
# PROGRESS PAGE
# ============================================================

elif st.session_state.page == "Progress":

    show_hero(
        "📈 Learning Progress",
        "See how consistently you are learning and testing yourself."
    )

    total_doubts = len(st.session_state.chat_history)
    total_quizzes = len(st.session_state.quiz_results)

    if total_quizzes:
        average = sum(
            r["score"] / r["total"] * 100
            for r in st.session_state.quiz_results
        ) / total_quizzes
    else:
        average = 0

    best = max(
        [r["score"] for r in st.session_state.quiz_results],
        default=0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Doubts Solved", total_doubts)

    with c2:
        st.metric("Quizzes", total_quizzes)

    with c3:
        st.metric("Average", f"{average:.0f}%")

    with c4:
        st.metric("Best Score", f"{best}/5")

    st.markdown("---")

    if not st.session_state.quiz_results:

        st.info(
            "Complete a quiz to start seeing your progress."
        )

    else:

        st.markdown("## 📝 Quiz Performance")

        for result in reversed(
            st.session_state.quiz_results
        ):

            percentage = (
                result["score"]
                / result["total"]
                * 100
            )

            st.markdown(
                f"**{result['topic']}** — "
                f"{result['score']}/{result['total']} "
                f"({percentage:.0f}%)"
            )

            st.progress(percentage / 100)


# ============================================================
# ABOUT PAGE
# ============================================================

elif st.session_state.page == "About":

    show_hero(
        "ℹ️ About the Project",
        "A student-focused AI learning assistant."
    )

    st.markdown(
        """
        <div class="classic-card">
        <h2>🧠 AI Doubt-to-Concept Tutor</h2>

        <p>
        This system is designed to help students understand the
        concept behind a doubt instead of simply giving an answer.
        </p>

        <h3>🔹 Main Features</h3>
        <ul>
            <li>AI-powered doubt explanation</li>
            <li>Concept identification</li>
            <li>Simple examples and real-world connections</li>
            <li>PDF-based study assistance</li>
            <li>Semantic search using embeddings</li>
            <li>ChromaDB knowledge storage</li>
            <li>Interactive 5-question quizzes</li>
            <li>Score and answer explanations</li>
            <li>Doubt history</li>
            <li>Learning progress dashboard</li>
        </ul>

        <h3>🛠️ Technologies</h3>
        <p>
        Python • Streamlit • Gemini • Sentence Transformers •
        ChromaDB • PyPDF
        </p>

        <h3>🔄 Basic Workflow</h3>
        <p>
        Student Doubt → Concept Detection → AI Explanation →
        Example → Quick Check → Quiz → Progress
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🧠 AI Doubt-to-Concept Tutor |
        Gemini + RAG + Sentence Transformers + ChromaDB
        <br>
        Learn • Understand • Practice • Improve
    </div>
    """,
    unsafe_allow_html=True
)