import streamlit as st
import faiss
import pickle
import numpy as np
import random
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Gemini API Key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

gemini = genai.GenerativeModel("gemini-2.5-flash")

# Load model and data
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("upsc_index.faiss")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Page settings
st.set_page_config(
    page_title="Kanishk's UPSC AI Mentor",
    page_icon="🇮🇳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}

.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #1e3a8a;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #475569;
    margin-bottom: 30px;
}

.welcome-card {
    background: linear-gradient(135deg,#ff9933,#ffffff,#138808);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.12);
}

.question-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.answer-box {
    background: #ffffff;
    padding: 25px;
    border-left: 6px solid #2563eb;
    border-radius: 12px;
    font-size: 17px;
    line-height: 1.7;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-title">
🇮🇳 Kanishk's UPSC AI Mentor
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Your journey to the Civil Services starts today
</div>
""", unsafe_allow_html=True)

# Welcome card
st.markdown("""
<div class="welcome-card">
<h2>Welcome, Kanishk 🇮🇳</h2>

<p style="font-size:20px;">
Every chapter you complete today is an investment in your future as a Civil Servant.
</p>

<p style="font-size:18px;">
📚 Study with consistency.<br>
🎯 Focus on the process.<br>
🏛️ Serve the nation with excellence.
</p>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.info("📖 NCERT-Based Answers")

with col2:
    st.success("⚡ Fast Concept Revision")

with col3:
    st.warning("🧠 AI-Powered Explanation")

# Random motivation
quotes = [
    "Consistency beats talent when talent doesn't work hard.",
    "Every NCERT chapter completed is a step towards UPSC success.",
    "Discipline today, IAS tomorrow.",
    "The pain of preparation is temporary. The pride of selection is forever.",
    "Stay focused. The nation needs dedicated civil servants.",
    "Dream big. Study hard. Serve India."
]

st.success("🌟 Today's Motivation: " + random.choice(quotes))

# Personal message
st.info("""
🌟 Dear Kanishk,

The road to UPSC is long, but every successful officer once sat where you are today.

There will be days when the syllabus feels endless.  
There will be days when results seem far away.

Keep learning.  
Keep improving.  
Keep believing.

Every NCERT chapter completed, every mock test attempted, and every hour of focused study is bringing you closer to your goal.

Success in UPSC is not about being the smartest.  
It is about being consistent every single day.

🇮🇳 Future Civil Servant in Progress 🇮🇳
""")

# Question section
st.markdown('<div class="question-card">', unsafe_allow_html=True)

question = st.text_input(
    "🔍 Ask your UPSC Question",
    placeholder="Example: What are Fundamental Rights?"
)

ask_button = st.button("🚀 Generate UPSC Answer")

st.markdown('</div>', unsafe_allow_html=True)

# Answer generation
if ask_button and question:
    with st.spinner("Searching NCERT books and preparing your UPSC answer..."):
        q_embedding = model.encode([question])

        D, I = index.search(
            np.array(q_embedding),
            k=5
        )

        context = ""

        for idx in I[0]:
            context += chunks[idx] + "\n\n"

        prompt = f"""
You are an expert UPSC mentor.

Use only the provided NCERT context.

Give the answer in this format:
1. Definition
2. Key Points
3. UPSC Exam Relevance
4. Short Conclusion

Context:
{context}

Question:
{question}

Answer:
"""

        response = gemini.generate_content(prompt)

    st.subheader("📚 UPSC Mentor's Answer")
    st.markdown(
        f'<div class="answer-box">{response.text}</div>',
        unsafe_allow_html=True
    )

elif ask_button and not question:
    st.error("Please enter a UPSC question first.")

# Footer
st.markdown("""
<hr>
<div class="footer">
<h4>🇮🇳 Kanishk's UPSC AI Mentor</h4>
<p>Built with dedication for a future Civil Servant</p>
<p>Dream • Study • Serve India</p>
</div>
""", unsafe_allow_html=True)