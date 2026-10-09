import os
import re
import time
import hashlib
import sqlite3
from io import BytesIO
from pathlib import Path

import streamlit as st
st.markdown("""
<style>
div.stButton:has(button p:contains("Process Medical Report")) button {
    background-color: #2196F3 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
}

div.stButton:has(button p:contains("Process Medical Report")) button:hover {
    background-color: #1976D2 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
from pypdf import PdfReader
import ollama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent

UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
DB_PATH = BASE_DIR / "users.db"

UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_MODEL = "qwen2.5:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION_STATE = {
    "logged_in": False,
    "user_id": None,
    "user_name": "",
    "username": "",
    "user_email": "",
    "vectorstore": None,
    "document_id": None,
    "document_name": None,
    "document_summary": "",
    "suggested_questions": [],
    "custom_question": "",
    "processed_file_hash": None
}

for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():
    connection = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """
    )

    connection.commit()

    return connection


# ============================================================
# PASSWORD HASHING
# ============================================================

def hash_password(password):
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# ============================================================
# REGISTER USER
# ============================================================

def register_user(name, username, email, password):

    connection = get_db_connection()

    try:
        connection.execute(
            """
            INSERT INTO users
            (name, username, email, password)
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                username,
                email,
                hash_password(password)
            )
        )

        connection.commit()

        return True, "Registration successful."

    except sqlite3.IntegrityError:

        return False, "Username or email already exists."

    finally:
        connection.close()


# ============================================================
# LOGIN USER
# ============================================================

def login_user(username_or_email, password):

    connection = get_db_connection()

    cursor = connection.execute(
        """
        SELECT
            id,
            name,
            username,
            email
        FROM users
        WHERE
            (username = ? OR email = ?)
            AND password = ?
        """,
        (
            username_or_email,
            username_or_email,
            hash_password(password)
        )
    )

    user = cursor.fetchone()

    connection.close()

    return user


# ============================================================
# CHECK OLLAMA
# ============================================================


def check_ollama():
    try:
        result = ollama.list()

        installed_models = []

        for model in result.models:
            installed_models.append(model.model)

        if OLLAMA_MODEL not in installed_models:
            return False, f"Ollama model '{OLLAMA_MODEL}' was not found."

        if EMBEDDING_MODEL not in installed_models:
            return False, f"Embedding model '{EMBEDDING_MODEL}' was not found."

        return True, "Ollama is ready."

    except Exception as e:
        return False, f"Ollama is not ready: {str(e)}"

# ============================================================
# GENERATE TEXT USING OLLAMA
# ============================================================

def generate_text(prompt, max_retries=2):

    for attempt in range(max_retries + 1):

        try:

            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.0,
                    "num_predict": 1200,
                    "num_ctx": 8192
                }
            )

            content = response["message"]["content"]

            return content.strip()

        except Exception as e:

            if attempt < max_retries:
                time.sleep(2)

            else:
                return f"ERROR: {str(e)}"


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes):

    try:

        pdf_stream = BytesIO(file_bytes)

        reader = PdfReader(pdf_stream)

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                text = page.extract_text()

                if text:

                    text = text.strip()

                    if text:

                        pages.append(
                            f"[Page {page_number}]\n{text}"
                        )

            except Exception:
                continue

        full_text = "\n\n".join(pages)

        return full_text.strip()

    except Exception as e:

        st.error(f"PDF reading error: {str(e)}")

        return ""


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vectorstore(chunks, document_id):

    embeddings = get_embeddings()

    collection_name = (
        "medical_report_" + document_id
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    try:

        existing = vectorstore.get()

        if existing and existing.get("ids"):

            vectorstore.delete(
                ids=existing["ids"]
            )

    except Exception:
        pass

    vectorstore.add_texts(chunks)

    return vectorstore


# ============================================================
# SPLIT DOCUMENT
# ============================================================

def split_document(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    return splitter.split_text(text)


# ============================================================
# GENERATE DOCUMENT SUMMARY
# ============================================================

def generate_document_summary(text):

    prompt = f"""
You are a medical report summarization assistant.

Read the medical report below and provide a concise summary
containing only the important information explicitly present
in the report.

DO NOT reproduce the entire report.
DO NOT copy the report sentence by sentence.
DO NOT invent information.
DO NOT use outside medical knowledge.

Use exactly these sections:

### Patient Information

Include important patient details such as:

- Name
- Age
- Gender
- Report date

If information is not present, write:
Not mentioned in the report.

### Important Test Results

Show important abnormal results first.

For each important result include:

- Test name
- Result
- Unit
- Reference range
- LOW / HIGH / NORMAL status

Mention important normal findings briefly.

Do not list every normal test.

### Findings / Impression

Summarize findings or impression explicitly stated
in the report.

### Medications

Mention medications only if present.

If medications are not mentioned, write:
Not mentioned in the report.

### Recommendations

Mention recommendations or follow-up instructions
only if explicitly present.

IMPORTANT RULES:

- Keep the summary concise.
- Preserve exact medical values.
- Preserve exact units.
- Preserve reference ranges.
- Preserve LOW, HIGH, or NORMAL status.
- Do not create a diagnosis.
- Do not use outside medical knowledge.
- Do not add treatment recommendations.
- If the report says "possible", preserve "possible".
- Use only information present in the report.

MEDICAL REPORT:

{text}
"""

    result = generate_text(
        prompt,
        max_retries=2
    )

    if result.startswith("ERROR:"):

        return (
            "Unable to generate the summary.\n\n"
            f"Technical error: {result}"
        )

    return result


# ============================================================
# GENERATE SUGGESTED QUESTIONS
# ============================================================

def generate_suggested_questions(summary):

    prompt = f"""
Generate exactly 6 useful questions that a user can ask
about this medical report.

Use ONLY information contained in the summary.

Focus on:

- abnormal test results
- important findings
- patient information
- reference ranges
- medications if mentioned
- recommendations

Do not introduce information not present in the summary.

Return ONLY six questions.

One question per line.

Do not number the questions.

SUMMARY:

{summary}
"""

    result = generate_text(
        prompt,
        max_retries=2
    )

    if result.startswith("ERROR:"):
        return []

    questions = []

    for line in result.splitlines():

        line = line.strip()

        if not line:
            continue

        line = re.sub(
            r"^[\s\-\*\d\.\)\:]+",
            "",
            line
        )

        line = line.strip()

        if line:
            questions.append(line)

    return questions[:6]


# ============================================================
# ANSWER QUESTION USING RAG
# ============================================================

def answer_question(question, vectorstore):

    fallback = (
        "I could not find this information "
        "in the uploaded report."
    )

    if vectorstore is None:
        return fallback

    try:

        docs = vectorstore.similarity_search(
            question,
            k=6
        )

    except Exception as e:

        return (
            "I could not search the uploaded report.\n\n"
            f"Technical error: {str(e)}"
        )

    if not docs:
        return fallback

    relevant_chunks = []

    for doc in docs:

        content = doc.page_content.strip()

        if (
            content
            and content not in relevant_chunks
        ):

            relevant_chunks.append(content)

    if not relevant_chunks:
        return fallback

    context = "\n\n---\n\n".join(
        relevant_chunks
    )

    prompt = f"""
You are a medical report question-answering assistant.

Answer the user's question using ONLY the report context.

Do not use outside medical information.
Do not guess.

If the answer cannot be found in the report,
respond exactly:

I could not find this information in the uploaded report.

Rules:

- Use exact values from the report.
- Include units when available.
- Include reference ranges when available.
- Mention LOW, HIGH, or NORMAL status when relevant.
- Do not invent information.
- Do not create a diagnosis that is not explicitly stated.
- Do not provide medical advice outside the report.
- Do not assume missing information.
- Clearly explain the answer.
- If the report says something is possible, keep that wording.

QUESTION:

{question}

REPORT CONTEXT:

{context}
"""

    answer = generate_text(
        prompt,
        max_retries=2
    )

    if answer.startswith("ERROR:"):
        return fallback

    return answer


# ============================================================
# SELECT SUGGESTED QUESTION
# ============================================================

def select_suggested_question(question):

    st.session_state.custom_question = question


# ============================================================
# AUTHENTICATION PAGE
# ============================================================

def show_auth_page():

    st.title("🏥 Medical RAG Assistant")

    st.markdown(
        """
        ### Analyze your medical reports using AI

        Upload a medical PDF report, view a concise
        summary, and ask questions about the report.
        """
    )

    st.divider()

    login_tab, register_tab = st.tabs(
        [
            "🔐 Login",
            "📝 Register"
        ]
    )

    # ========================================================
    # LOGIN
    # ========================================================

    with login_tab:

        st.subheader("Login")

        login_username = st.text_input(
            "Username or Email",
            key="login_username"
        )

        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        login_clicked = st.button(
            "Login",
            use_container_width=True
        )

        if login_clicked:

            if (
                not login_username.strip()
                or not login_password
            ):

                st.warning(
                    "Please enter username/email and password."
                )

            else:

                user = login_user(
                    login_username.strip(),
                    login_password
                )

                if user:

                    (
                        user_id,
                        name,
                        username,
                        email
                    ) = user

                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.user_name = name or ""
                    st.session_state.username = username or ""
                    st.session_state.user_email = email or ""

                    st.success(
                        "Login successful."
                    )

                    time.sleep(0.5)

                    st.rerun()

                else:

                    st.error(
                        "Invalid username/email or password."
                    )

    # ========================================================
    # REGISTER
    # ========================================================

    with register_tab:

        st.subheader("Create Account")

        register_name = st.text_input(
            "Name",
            key="register_name"
        )

        register_username = st.text_input(
            "Username",
            key="register_username"
        )

        register_email = st.text_input(
            "Email",
            key="register_email"
        )

        register_password = st.text_input(
            "Password",
            type="password",
            key="register_password"
        )

        register_confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="register_confirm_password"
        )

        register_clicked = st.button(
            "Register",
            use_container_width=True
        )

        if register_clicked:

            if not register_name.strip():

                st.warning(
                    "Please enter your name."
                )

            elif not register_username.strip():

                st.warning(
                    "Please enter a username."
                )

            elif not register_email.strip():

                st.warning(
                    "Please enter your email."
                )

            elif not register_password:

                st.warning(
                    "Please enter a password."
                )

            elif (
                register_password
                != register_confirm_password
            ):

                st.error(
                    "Passwords do not match."
                )

            else:

                success, message = register_user(
                    register_name.strip(),
                    register_username.strip(),
                    register_email.strip(),
                    register_password
                )

                if success:

                    st.success(message)

                else:

                    st.error(message)


# ============================================================
# SIDEBAR
# ============================================================

def show_sidebar():

    with st.sidebar:

        st.title("🏥 Medical RAG")

        st.divider()

        user_name = (
            st.session_state.user_name
            or st.session_state.username
            or "User"
        )

        st.write(
            f"👋 Welcome, **{user_name}**"
        )

        if st.session_state.user_email:

            st.write(
                f"📧 {st.session_state.user_email}"
            )

        st.divider()

        st.markdown(
            """
            ### Features

            📄 Upload medical PDF

            📋 Concise report summary

            🔎 Search report information

            💬 Ask questions

            🤖 Local AI using Ollama

            🔐 Local user authentication
            """
        )

        st.divider()

        if st.button(
            "🚪 Logout",
            use_container_width=True
        ):

            for key, value in DEFAULT_SESSION_STATE.items():
                st.session_state[key] = value

            st.rerun()


# ============================================================
# PROCESS MEDICAL REPORT
# ============================================================

def process_medical_report(
    uploaded_file,
    file_bytes,
    file_hash
):

    # ========================================================
    # CHECK OLLAMA
    # ========================================================

    with st.spinner("🤖 Checking Ollama..."):

        ollama_ready, ollama_message = check_ollama()

    if not ollama_ready:

        st.error("⚠️ Ollama is not ready.")

        st.error(ollama_message)

        st.info(
            "Make sure Ollama is running and both models are installed."
        )

        st.code(
            "ollama serve\n"
            "ollama list"
        )

        return

    # ========================================================
    # SAVE PDF
    # ========================================================

    safe_filename = Path(
        uploaded_file.name
    ).name

    save_path = UPLOAD_DIR / safe_filename

    try:

        with open(
            save_path,
            "wb"
        ) as file:

            file.write(file_bytes)

    except Exception as e:

        st.error(
            f"Error saving PDF: {str(e)}"
        )

        return

    # ========================================================
    # EXTRACT TEXT
    # ========================================================

    with st.spinner(
        "📖 Reading medical report..."
    ):

        full_text = extract_pdf_text(
            file_bytes
        )

    if not full_text:

        st.error(
            "❌ No readable text was found in this PDF."
        )

        st.warning(
            "This may be a scanned/image-only PDF. "
            "The current application works with text-based PDFs."
        )

        return

    st.success(
        f"✅ PDF text extracted successfully "
        f"({len(full_text):,} characters)."
    )

    # ========================================================
    # DOCUMENT ID
    # ========================================================

    document_id = hashlib.md5(
        full_text.encode("utf-8")
    ).hexdigest()[:16]

    # ========================================================
    # SPLIT DOCUMENT
    # ========================================================

    with st.spinner(
        "🔎 Preparing report for question answering..."
    ):

        chunks = split_document(
            full_text
        )

    if not chunks:

        st.error(
            "Could not create document chunks."
        )

        return

    # ========================================================
    # CREATE VECTOR DATABASE
    # ========================================================

    try:

        with st.spinner(
            "🧠 Creating medical report search index..."
        ):

            vectorstore = create_vectorstore(
                chunks,
                document_id
            )

    except Exception as e:

        st.error(
            "❌ Error creating vector database."
        )

        st.error(
            str(e)
        )

        return

    # ========================================================
    # GENERATE SUMMARY
    # ========================================================

    with st.spinner(
        "📋 Preparing important information from the report..."
    ):

        summary = generate_document_summary(
            full_text
        )

    # ========================================================
    # GENERATE QUESTIONS
    # ========================================================

    with st.spinner(
        "💡 Preparing useful questions..."
    ):

        suggested_questions = (
            generate_suggested_questions(
                summary
            )
        )

    # ========================================================
    # SAVE SESSION STATE
    # ========================================================

    st.session_state.vectorstore = vectorstore

    st.session_state.document_id = document_id

    st.session_state.document_name = (
        uploaded_file.name
    )

    st.session_state.document_summary = summary

    st.session_state.suggested_questions = (
        suggested_questions
    )

    st.session_state.custom_question = ""

    st.session_state.processed_file_hash = (
        file_hash
    )

    st.success(
        "✅ Medical report processed successfully."
    )

    time.sleep(0.5)

    st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

def show_dashboard():

    show_sidebar()

    # ========================================================
    # HEADER
    # ========================================================

    st.title(
        "🏥 Medical RAG Assistant"
    )

    st.markdown(
        """
        Upload a medical report and get a concise summary
        of the important information. You can then ask
        questions about the uploaded report.
        """
    )

    # ========================================================
    # UPLOAD SECTION
    # ========================================================

    st.divider()

    st.subheader(
        "📄 Upload Medical Report"
    )

    uploaded_file = st.file_uploader(
        "Choose a medical PDF report",
        type=["pdf"],
        key="medical_pdf",
        help="Upload a text-based medical PDF."
    )

    if uploaded_file is not None:

        file_bytes = uploaded_file.getvalue()

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()

        st.success(
            f"📄 Selected: {uploaded_file.name}"
        )

        st.caption(
            f"File size: {len(file_bytes) / 1024:.1f} KB"
        )

        # ====================================================
        # PROCESS BUTTON
        # ====================================================

        if (
            st.session_state.processed_file_hash
            != file_hash
        ):

            process_clicked = st.button(
                "⚙️ Process Medical Report",
                type="secondary",
                use_container_width=True
            )

            if process_clicked:

                process_medical_report(
                    uploaded_file,
                    file_bytes,
                    file_hash
                )

        else:

            st.success(
                "✅ This report has already been processed."
            )

    else:

        st.info(
            "Please select a PDF medical report."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    if st.session_state.document_summary:

        st.divider()

        st.subheader(
            "📋 Summary"
        )

        st.markdown(
            st.session_state.document_summary
        )

    # ========================================================
    # ASK QUESTIONS
    # ========================================================

    if st.session_state.vectorstore is not None:

        st.divider()

        st.subheader(
            "❓ Ask Questions About Your Report"
        )

        st.markdown(
            "**Ask your question:**"
        )

        question_col, search_col = st.columns(
            [6, 1],
            gap="small"
        )

        # ====================================================
        # QUESTION INPUT
        # ====================================================

        with question_col:

            st.text_input(
                "Question",
                placeholder=(
                    "Example: What are the abnormal test results?"
                ),
                key="custom_question",
                label_visibility="collapsed"
            )

        # ====================================================
        # SEARCH BUTTON
        # ====================================================

        with search_col:

            search_clicked = st.button(
                "🔍 Search",
                key="search_question_button",
                use_container_width=True
            )

        # ====================================================
        # SUGGESTED QUESTIONS
        # ====================================================

        questions = (
            st.session_state.suggested_questions
        )

        if questions:

            st.markdown(
                "**💡 Questions you may want to ask:**"
            )

            for i, question in enumerate(questions):

                st.button(
                    question,
                    key=f"suggested_question_{i}",
                    use_container_width=True,
                    on_click=select_suggested_question,
                    args=(question,)
                )

        # ====================================================
        # SEARCH ANSWER
        # ====================================================

        if search_clicked:

            question_to_search = (
                st.session_state.custom_question.strip()
            )

            if not question_to_search:

                st.warning(
                    "Please type a question or select one of the questions below."
                )

            else:

                with st.spinner(
                    "🔎 Searching the report..."
                ):

                    answer = answer_question(
                        question_to_search,
                        st.session_state.vectorstore
                    )

                st.markdown(
                    "### 💬 Answer"
                )

                st.write(answer)


# ============================================================
# MAIN APPLICATION
# ============================================================

if not st.session_state.logged_in:

    show_auth_page()

else:

    show_dashboard()