# 🚀 [Tips Hindawi](https://www.tipshindawi.com/) Internship (August–October) 2026

> 🎓 This project was built during the [ **Tips Hindawi** ](https://www.tipshindawi.com/) **Internship (August–October) 2026**.

## 👤 Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        | Mohammad Prince Abdelrahman       |
| Project Name     | Study Chat Bot                            |
| GitHub Username  | MohammadPrince2004               |
| Internship Batch | August–October 2026                  |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [**Edrak for Ai**](https://edrak4ai.com/en)                         |

---

# 📖 Project Overview

**Study Bot** is an AI study assistant that turns any lecture PDF into a **structured summary**, a **chat that answers only from the lecture** (RAG), and an **auto-generated quiz** with scoring and a per-question review.

The goal was to apply the topics covered in the training (Transformers, Hugging Face, RAG, and LangChain with chains and output parsers) in one real, usable application, with a Streamlit interface on top.

---

# ✨ Features

* 📝 **Summary:** bullet-point summary of long lectures using a Map-Reduce approach.
* 💬 **Chat with your lecture (RAG):** answers are grounded only in the uploaded file, with the source passages shown. If the answer is not in the lecture, the bot says so instead of guessing.
* 🧠 **Quiz:** multiple-choice questions generated from the summary and shown one at a time.
* 📊 **Result & review:** score, percentage, grade, and a review of every question with the correct answer and an explanation.

---

# 🛠️ Technologies Used

| Part | Tool |
|---|---|
| LLM | `Qwen/Qwen2.5-3B-Instruct` with Hugging Face Transformers (`text-generation` pipeline) |
| Orchestration | LangChain: prompt templates, chains (LCEL), output parsers |
| Retrieval | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` + FAISS |
| Interface | Streamlit |
| PDF parsing | pypdf |
| Environment | Python, PyTorch, Kaggle GPU (T4) |

**How it works:**

```
PDF ──► text ──► split into chunks ──┬──► Map-Reduce summary ──► Quiz (JSON parser + validation)
                                     └──► embeddings ──► FAISS ──► retrieve top-k ──► LLM answer (RAG)
```

---

# ⚙️ Installation

**Requirements:** Python 3.10+ and a GPU (recommended). The 3B model in `float16` needs roughly 6 GB of GPU memory.

```bash
git clone https://github.com/MohammadPrince2004/Study-Chat-Bot.git
cd Study-Chat-Bot
pip install -r requirements.txt
```

The first run downloads the models from Hugging Face (about 6 GB for the LLM and 470 MB for the embeddings), so an internet connection is needed.

**Running on Kaggle:** enable **Internet** and a **GPU (T4)**, save `app.py` with `%%writefile app.py`, run Streamlit in the background, and expose it with a tunnel such as ngrok.

---

# 🚀 Usage

```bash
streamlit run app.py
```

1. Upload a lecture PDF and click **Summarize**.
2. Read the **Summary** tab, or open the **Chat** tab and ask questions about the lecture.
3. Open the **Quiz** tab, choose the number of questions, and click **Start quiz**.
4. Answer the questions one by one and check your **score and review** at the end.

---

# 📸 Demo

Add screenshots, GIFs, or a demo video here.

```
docs/upload.png
docs/summary.png
docs/chat.png
docs/quiz.png
docs/quiz_1.png
docs/evaluation.png
```

---

# 📈 Results

* Built a complete pipeline in one app: **summarize → chat → quiz → evaluation**.
* Reduced made-up answers in the chat by restricting the model to retrieved lecture passages and showing the sources.
* Made a small 3B model return usable quiz data by requesting JSON, **validating** every question (4 options, answer must be one of them), and **retrying** on failure.
* Handled long lectures with Map-Reduce summarization and chunking.
* Tested on a real university lecture (Operating Systems & FreeRTOS slides).

**Known limitations:** a 3B model can still make mistakes, quiz questions come from the summary so they cover the main points rather than every detail, scanned PDFs need OCR, and only the first few chunks are summarized by default (`max_parts=3`) to keep it fast.

---

# 🔮 Future Improvements

* Flashcards generated from the lecture
* Re-take only the questions answered incorrectly
* Save scores and track progress over time
* Conversation memory in the chat
* OCR support for scanned lectures

---

# 📚 About the Internship

This project was developed as part of the [**Tips Hindawi**](https://www.tipshindawi.com/) **Internship (August–October) 2026**, and it will be showcased on the official [Tips Hindawi](https://www.tipshindawi.com/) website.

[Tips Hindawi](https://www.tipshindawi.com/) is the internships department of [**Edrak for Ai**](https://edrak4ai.com/en), and the internship encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

For more information about the internship, training programs, and upcoming batches, visit the official [Tips Hindawi](https://www.tipshindawi.com/) website.

---

# 📄 License

This project is shared for educational and portfolio purposes.
