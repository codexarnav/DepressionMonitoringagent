
import os
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import pipeline
from datetime import datetime
from typing import TypedDict, Sequence, Annotated, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    transcribed_text: Annotated[Optional[str], None]
    detected_emotion: Annotated[Optional[str], None]
    problem_category: Annotated[Optional[str], None]
    suggested_action: Annotated[Optional[str], None]
    next_checkin_time: Annotated[Optional[datetime], None]
    is_emergency: Annotated[bool, None]

import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="input.wav", duration=5, fs=16000):  # 16kHz recommended for Whisper
    print("Recording... Speak Now!")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print("Recording saved as", filename)



def enter_input(state: AgentState) -> AgentState:
    """User enters the input through text or live microphone"""
    choice = input("Choose input method (1 for Text, 2 for Voice): ")

    if choice == '1':
        user_input = input('Enter how you feel: ')
    elif choice == '2':
        record_audio()  # records and saves as 'input.wav'
        model = whisper.load_model("base")
        result = model.transcribe("input.wav")
        user_input = result['text']
        print(f"Transcribed Text: {user_input}")
    else:
        print("Invalid choice. Defaulting to text input.")
        user_input = input('Enter how you feel: ')

    state['transcribed_text'] = user_input
    return state


def sentiment_analysis(state: AgentState) -> AgentState:
    """Classify user input emotion and detect emergencies"""
    classifier = pipeline("sentiment-analysis")
    user_input = state['transcribed_text']
    
    result = classifier(user_input)[0]
    emotion = result['label']  

    if emotion == "POSITIVE":
        category = "positive mood"
    elif emotion == "NEGATIVE":
        category = "stress or sadness"
    else:
        category = "neutral"

    emergency_keywords = ["suicide", "kill myself", "end my life", "hurt myself", "no way out", "giving up"]
    is_emergency = any(keyword in user_input.lower() for keyword in emergency_keywords)

    if is_emergency:
        print("⚠️ Emergency situation detected! Immediate attention required.")

    state['detected_emotion'] = emotion
    state['problem_category'] = category
    state['is_emergency'] = is_emergency
    return state

llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.1, api_key='AIzaSyBlVAGaEchXF46DbcNFE66X6HbbcN4oSXI')

def chat_model(state: AgentState) -> AgentState:
    """Provide calming suggestions or emergency resources"""
    problem = state['problem_category']

    if state['is_emergency']:
        emergency_msg = ("⚠️ It seems you're in distress. "
                         "Please immediately reach out to a helpline: 1800-599-0019 (India Suicide Prevention) "
                         "or talk to a trusted person nearby.")
        print(emergency_msg)
        state['suggested_action'] = emergency_msg
        return state

    system_prompt = SystemMessage(
        content="You are a helpful therapist who suggests simple positive actions based on a user's mood or problem."
    )
    user_message = HumanMessage(
        content=f"The user is feeling: {problem}. Suggest 3 simple, practical actions they can take right now to feel better."
    )

    response = llm.invoke([system_prompt, user_message])
    suggested_action = response.content
    print("Suggested Action:", suggested_action)

    state['suggested_action'] = suggested_action
    return state


graph = StateGraph(AgentState)
graph.add_node('input', enter_input)
graph.add_node('sentiment', sentiment_analysis)
graph.add_node('llm', chat_model)

graph.add_edge(START, 'input')
graph.add_edge('input', 'sentiment')
graph.add_edge('sentiment', 'llm')
graph.add_edge('llm', END)

app = graph.compile()

# --- Run the Agent ---
if __name__ == "__main__":
    # Initial State
    inputs = {
        'messages': [],
        'transcribed_text': None,
        'detected_emotion': None,
        'problem_category': None,
        'suggested_action': None,
        'next_checkin_time': None,
        'is_emergency': False
    }

    # Run the graph
    output = app.invoke(inputs)
    print("\n=== Final Agent Output ===")
    print(output)
