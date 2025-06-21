# 🧠 Mental Health AI Agent with Emergency SMS 🚨

A Streamlit-based AI-powered Mental Health Assistant that:
- Accepts **voice or text input**.
- Detects **emotions** using a lightweight NLP model.
- Provides **personalized mood improvement suggestions** using Google's Gemini LLM.
- Detects signs of **emergency/distress (suicidal intent)** and auto-sends an **SMS alert to an emergency contact via Twilio**.
- Displays **suicide prevention helpline information** if a crisis is detected.

---

## 🔍 **Features**

✅ Real-time **voice or text-based interaction**  
✅ **Speech-to-text** transcription using Google Speech Recognition  
✅ **Emotion detection** using `cardiffnlp/twitter-roberta-base-sentiment-latest` model  
✅ **Personalized suggestions** powered by Gemini AI  
✅ Automatic **emergency SMS alert** to registered contact via Twilio  
✅ Displays **global suicide prevention helplines** for immediate assistance  

---

## 🏗️ **Tech Stack**

| Tool | Purpose |
|------|---------|
| **Python** | Core programming language |
| **Streamlit** | Web UI |
| **Google SpeechRecognition** | Speech to text |
| **Hugging Face Transformers** | Sentiment Analysis |
| **Langchain + Gemini (Google GenAI)** | LLM-based suggestions |
| **Twilio** | Emergency SMS sending |
| **LangGraph** | State machine handling AI flow |

---

## 🚀 **Quick Start**

### 1. **Clone Repo**
```bash
git clone https://github.com/your-username/mental-health-ai-agent.git
cd mental-health-ai-agent
pip install -r requirements.txt
streamlit run app.py
