## 📁 Project Structure

text
medical-rag-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore


### Main File

#### `app.py`

Contains the complete application:

* User registration and login
* PDF upload
* PDF text extraction
* Text chunking
* Nomic text embeddings
* ChromaDB vector storage
* Medical report summarization
* Suggested question generation
* RAG-based question answering
* Streamlit user interface

### Local/Generated Files

The following files and folders are created locally while using the application and are excluded from GitHub using `.gitignore`:

```text
users.db
uploads/
chroma_db/
venv/
.env
```

* `users.db` — SQLite database used for local user authentication.
* `uploads/` — Stores uploaded medical PDF reports.
* `chroma_db/` — Stores the locally generated ChromaDB vector data.
* `venv/` — Python virtual environment.
* `.env` — Local environment configuration, if required.

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/JuvvanapudiSrujana/medical-rag-assistant.git
```

Move into the project directory:

bash
cd medical-rag-assistant


---

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

---

### 3. Install Required Packages

Install the dependencies from the project's requirements file:

```powershell
pip install -r requirements.txt
```

---

## 🤖 Ollama Setup

Install Ollama on your system.

Then download the required models.

### Qwen 2.5

```powershell
ollama pull qwen2.5
```

### Nomic Embeddings

```powershell
ollama pull nomic-embed-text
```

Check the installed models:

```powershell
ollama list
```

You should see models similar to:

```text
qwen2.5:latest
nomic-embed-text:latest
```

---

## ▶️ Running the Application

Ollama must be running before starting the Streamlit application.

### Terminal 1

Open PowerShell and run:

```powershell
ollama serve
```

Keep this terminal running.

### Terminal 2

Open another PowerShell window:

```powershell
cd C:\Users\Srujana\medical-rag-assistant
```

Activate the virtual environment:

```powershell
venv\Scripts\activate
```

Run the application:

```powershell
streamlit run app.py
```

The application will open in your browser.

---

## 🔑 Application Flow

### Step 1 — Register

Create an account using:

text
Name
Username
Email
Password
Confirm Password
```

### Step 2 — Login

Login using:

```text
Username or Email
Password
```

### Step 3 — Upload Report

Upload a medical PDF report and process it using the application.

### Step 4 — View Summary

The application generates a concise summary containing important information extracted from the medical report.

### Step 5 — Ask Questions

Enter a question in:

```text
❓ Ask Questions About Your Report
```

Then click:

```text
🔍 Search
```

### Step 6 — View Answer

The RAG system retrieves relevant information from the uploaded report and generates a report-based answer.
