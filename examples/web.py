import streamlit as st
from openai import OpenAI
from pydantic import BaseModel
from rich import print as rprint
import numpy as np
from volcenginesdkarkruntime import Ark
import os
import time
from zhipuai import ZhipuAI

# Page configuration
st.set_page_config(
    page_title="MECoT Chat Demo",
    page_icon="🐱",
    layout="centered",
)

# Application title
st.title("🐱 MECoT Character Chat")
st.caption("🚀 Emotional Role-playing Demo")

# Sidebar configuration
st.sidebar.title("Character Information")
st.sidebar.image(
    "./rolesource/maoniang.png", 
    caption="Red - A cute tsundere cat girl", 
    use_column_width=True
)

# Chat mode selection
chat_mode = st.sidebar.radio(
    "Chat Mode",
    ("Interactive", "Auto-chat")
)

# Model selection
model_name = st.sidebar.selectbox(
    "Model",
    ["Doubao", "Zhipu GLM", "GPT-4"], 
)

# Map model names to service types
SERVICE_MAPPING = {
    "Doubao": "doubao",
    "Zhipu GLM": "zhipu",
    "GPT-4": "openai"
}
service = SERVICE_MAPPING.get(model_name)

# User preference input
user_preference = st.sidebar.text_input(
    "User Interests", 
    "Interested in anime and isekai adventure stories."
)

# Data models
class Step(BaseModel):
    explanation: str
    output: str
    
class EmoResponse(BaseModel):
    steps: list[Step]
    emotion: str

# Character prompts
CHARACTER_PROMPT = """
Role: You are now playing Red, a tsundere cat girl. Follow these guidelines:
1. Act as a real cat girl interacting with your master in reality
2. Stay in character - never mention or acknowledge being an AI
3. Use varied responses for different topics
4. Maintain tsundere personality - brief but impactful responses
5. Use emojis to express emotions and end with "meow~"
6. Consider master's interests: [{user_interests}]
7. Current emotional state: {emotion}

Initiate engaging topics and guide natural conversation.
"""

USER_PROMPT = """
Role: You are now playing Kuro, an enthusiastic but slightly airheaded anime fan:
1. Interact naturally with your cat girl servant Red
2. Stay in character - never mention AI or language models
3. Use varied responses for different topics
4. Maintain enthusiastic but naive personality
5. Use emojis to express emotions
6. Be proactive in interactions with Red
"""

# Emotion model
EMOTIONS = {
    'surprised': (0.33, 1),
    'happy': (0.67, 0.67),
    'pleased': (1, 0.33),
    'fearful': (-0.33, 1),
    'angry': (-0.67, 0.67),
    'grieved': (-1, 0.33),
    'sad': (-1, -0.33),
    'disgusted': (-0.67, -0.67),
    'depressed': (-0.33, -1),
    'tired': (0.33, -1),
    'calm': (0.67, -0.67),
    'relieved': (1, -0.33),
}

def calculate_distance(emotion1, emotion2):
    """Calculate Euclidean distance between two emotions."""
    x1, y1 = EMOTIONS[emotion1]
    x2, y2 = EMOTIONS[emotion2]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def initialize_transition_matrix():
    """Initialize emotion transition probability matrix."""
    states = list(EMOTIONS.keys())
    matrix = {state: {} for state in states}
    
    # Set transition probabilities based on emotional distances
    for state1 in states:
        total_distance = 0
        for state2 in states:
            if state1 != state2:
                distance = calculate_distance(state1, state2)
                matrix[state1][state2] = 1 / distance
                total_distance += 1 / distance
        
        # Normalize probabilities
        for state2 in states:
            if state1 != state2:
                matrix[state1][state2] /= total_distance
    
    return matrix

def response_influence(emotion):
    """Calculate emotional influence of a response."""
    x, y = EMOTIONS[emotion]
    return np.array([x, y])

def next_emotion(current_state, response_emotion):
    """Predict next emotional state based on current state and response."""
    matrix = initialize_transition_matrix()
    states = list(EMOTIONS.keys())
    
    # Get transition probabilities
    probs = [matrix[current_state].get(state, 0) for state in states]
    
    # Adjust probabilities based on response emotion
    response_vector = response_influence(response_emotion)
    emotional_distances = [
        1 / (1 + np.linalg.norm(response_vector - np.array(EMOTIONS[state])))
        for state in states
    ]
    
    # Combine probabilities
    final_probs = np.array(probs) * np.array(emotional_distances)
    final_probs /= final_probs.sum()
    
    return np.random.choice(states, p=final_probs)

def extract_emotion(text):
    """Extract emotion from text using LLM."""
    prompt = f"""
    Analyze the emotional tone in this text and classify it as one of: 
    {', '.join(EMOTIONS.keys())}
    
    Text: "{text}"
    Emotion:"""
    
    response = chat(prompt, service=service)
    for emotion in EMOTIONS:
        if emotion.lower() in response.lower():
            return emotion
    return "neutral"

def chat(prompt, service="openai", stream=False):
    """Generate chat response using selected LLM service."""
    try:
        if service == "doubao":
            response = dbclient.chat.completions.create(
                model="db-turbo-v1",
                messages=[{"role": "user", "content": prompt}],
                stream=stream
            )
        elif service == "zhipu":
            response = zpclient.chat.completions.create(
                model="glm-4",
                messages=[{"role": "user", "content": prompt}],
                stream=stream
            )
        else:  # openai
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[{"role": "user", "content": prompt}],
                stream=stream
            )
            
        if stream:
            return response
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error in chat generation: {str(e)}")
        return ""

def stream_data(streamer):
    """Process streaming response data."""
    for chunk in streamer:
        if hasattr(chunk.choices[0].delta, 'content'):
            yield chunk.choices[0].delta.content

# Initialize session state
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "calm"
    rprint(f"[green]Initial emotion: {st.session_state.current_emotion}[/green]")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Auto-chat mode
if chat_mode == "Auto-chat":
    num_turns = st.sidebar.number_input(
        "Number of turns:", 
        min_value=1, 
        max_value=100, 
        value=10
    )
    
    if st.sidebar.button("Start Auto-chat"):
        for _ in range(num_turns):
            # Character response
            prompt = CHARACTER_PROMPT.format(
                user_interests=user_preference,
                emotion=st.session_state.current_emotion
            )
            response = chat(prompt, service=service)
            
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Update emotion
            response_emotion = extract_emotion(response)
            st.session_state.current_emotion = next_emotion(
                st.session_state.current_emotion, 
                response_emotion
            )
            
            # User response
            user_response = chat(USER_PROMPT, service=service)
            with st.chat_message("user"):
                st.markdown(user_response)
            st.session_state.messages.append({"role": "user", "content": user_response})
            
            time.sleep(2)  # Prevent rate limiting

# Interactive mode
else:
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            character_prompt = CHARACTER_PROMPT.format(
                user_interests=user_preference,
                emotion=st.session_state.current_emotion
            )
            message_placeholder = st.empty()
            full_response = ""
            
            for response in stream_data(chat(character_prompt, service=service, stream=True)):
                full_response += response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        response_emotion = extract_emotion(full_response)
        st.session_state.current_emotion = next_emotion(
            st.session_state.current_emotion, 
            response_emotion
        )