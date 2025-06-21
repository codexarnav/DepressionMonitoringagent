import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Sequence, Annotated, Optional
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from twilio.rest import Client
import tempfile
import os
from datetime import datetime

# ------------------- TWILIO SETUP -------------------------
TWILIO_ACCOUNT_SID = 'AC1b8b3442f9434e8381abb00be3da8642'
TWILIO_AUTH_TOKEN = '73f13596e5a19fb922df96c70dbe7d84'
TWILIO_PHONE_NUMBER = '+919625984260'  # eg: '+1234567890'
EMERGENCY_CONTACT_NUMBER = ''  # eg: '+9199xxxxxxxx'

def send_emergency_sms(message):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=EMERGENCY_CONTACT_NUMBER
        )
        st.success("🚨 Emergency SMS sent to contact!")
    except Exception as e:
        st.error(f"❌ SMS sending failed: {e}")

# ------------------- AGENT STATE -------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    transcribed_text: Optional[str]
    detected_emotion: Optional[str]
    problem_category: Optional[str]
    suggested_action: Optional[str]
    next_checkin_time: Optional[datetime]
    is_emergency: Optional[bool]

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

@st.cache_resource
def load_speech_recognizer():
    return sr.Recognizer()

classifier = load_sentiment_pipeline()
recognizer = load_speech_recognizer()

try:
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0.1,
        api_key='Your_api_key'
    )
except Exception as e:
    st.error(f"Error loading Gemini: {e}")
    llm = None

# ------------------- NODES -------------------------
def input_node(state: AgentState) -> AgentState:
    if 'audio_bytes' in st.session_state and st.session_state.audio_bytes:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(st.session_state.audio_bytes)
                temp_path = tmp_file.name

            with sr.AudioFile(temp_path) as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.record(source)
                transcribed = recognizer.recognize_google(audio)
                state['transcribed_text'] = transcribed
                st.success(f"✅ Transcribed: {transcribed}")

            os.remove(temp_path)

        except Exception as e:
            st.error(f"Transcription error: {e}")
            state['transcribed_text'] = ""

    elif 'text_input' in st.session_state and st.session_state.text_input:
        state['transcribed_text'] = st.session_state.text_input

    return state

def sentiment_node(state: AgentState) -> AgentState:
    text = state['transcribed_text']
    if text:
        result = classifier(text)[0]
        emotion = result['label']
        state['detected_emotion'] = emotion

        if emotion == "positive":
            state['problem_category'] = "positive mood"
        elif emotion == "negative":
            state['problem_category'] = "stress or sadness"
        else:
            state['problem_category'] = "neutral"

        emergency_keywords = ["suicide", "kill myself", "end my life", "hurt myself", "giving up"]
        state['is_emergency'] = any(k in text.lower() for k in emergency_keywords)
    return state

def llm_node(state: AgentState) -> AgentState:
    if not llm:
        state['suggested_action'] = "Gemini model unavailable."
        return state

    problem = state['problem_category']
    system_prompt = SystemMessage(content="You are a helpful therapist suggesting calming actions.")
    user_msg = HumanMessage(content=f"The user is feeling: {problem}. Suggest 3 simple actions.")

    try:
        response = llm.invoke([system_prompt, user_msg])
        state['suggested_action'] = response.content
    except Exception as e:
        st.error(f"Gemini error: {e}")
        state['suggested_action'] = "Default tips: Breathe, walk, talk to a friend."

    return state

# ------------------- LANGGRAPH SETUP -------------------------
graph = StateGraph(AgentState)
graph.add_node('input', input_node)
graph.add_node('sentiment', sentiment_node)
graph.add_node('llm', llm_node)
graph.add_edge(START, 'input')
graph.add_edge('input', 'sentiment')
graph.add_edge('sentiment', 'llm')
graph.add_edge('llm', END)
app = graph.compile()

# ------------------- STREAMLIT UI -------------------------
st.title("🎙️ Mental Health AI Agent with Emergency SMS")

col1, col2 = st.columns(2)

# -------- Voice Input --------
with col1:
    st.subheader("🎤 Voice Input")

    audio_file = st.audio_input("Record your voice message here:")
    if audio_file is not None:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")

        if st.button("Analyze Voice"):
            st.session_state.audio_bytes = audio_bytes
            st.session_state.text_input = None

            state = {
                'messages': [],
                'transcribed_text': None,
                'detected_emotion': None,
                'problem_category': None,
                'suggested_action': None,
                'next_checkin_time': None,
                'is_emergency': False
            }

            with st.spinner("Processing voice..."):
                output = app.invoke(state)

                st.markdown("### 📋 Results")
                st.write(f"📝 **Transcribed Text:** {output['transcribed_text']}")
                st.write(f"😊 **Detected Emotion:** {output['detected_emotion']}")
                st.write(f"🔍 **Category:** {output['problem_category']}")
                st.write(f"🚨 **Emergency:** {'Yes' if output['is_emergency'] else 'No'}")
                st.write(f"💡 **AI Suggestions:** {output['suggested_action']}")

                if output['is_emergency']:
                    st.error("⚠️ EMERGENCY DETECTED!")
                    st.warning("📞 Suicide Prevention Helpline: 1800-599-0019 (India) or 988 (USA)")
                    
                    # 🚨 Send SMS
                    send_emergency_sms("⚠️ Emergency detected! The user may be in danger. Please check immediately.")

# -------- Text Input --------
with col2:
    st.subheader("✍️ Text Input")
    text_input = st.text_area("How are you feeling?", height=150)

    if st.button("Analyze Text"):
        if text_input.strip():
            st.session_state.text_input = text_input
            st.session_state.audio_bytes = None

            state = {
                'messages': [],
                'transcribed_text': None,
                'detected_emotion': None,
                'problem_category': None,
                'suggested_action': None,
                'next_checkin_time': None,
                'is_emergency': False
            }

            with st.spinner("Processing text..."):
                output = app.invoke(state)

                st.markdown("### 📋 Results")
                st.write(f"📝 **Input Text:** {output['transcribed_text']}")
                st.write(f"😊 **Detected Emotion:** {output['detected_emotion']}")
                st.write(f"🔍 **Category:** {output['problem_category']}")
                st.write(f"🚨 **Emergency:** {'Yes' if output['is_emergency'] else 'No'}")
                st.write(f"💡 **AI Suggestions:** {output['suggested_action']}")

                if output['is_emergency']:
                    st.error("⚠️ EMERGENCY DETECTED!")
                    st.warning("📞 Suicide Prevention Helpline: 1800-599-0019 (India) or 988 (USA)")
                    
                    # 🚨 Send SMS
                    send_emergency_sms("⚠️ Emergency detected! The user may be in danger. Please check immediately.")
