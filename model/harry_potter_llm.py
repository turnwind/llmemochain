from . import llmservice
from .emotion import EmotionalAgent, EmotionExtractor, EmoResponse, Step
import os
import numpy as np

class HarryPotterLLM:
    def __init__(self, service_type="doubao"):
        # Initialize emotional model
        # Harry's personality traits: brave (more prone to anger and surprise), loyal (more prone to happiness and sadness)
        self.personality_weights = [
            3,  # surprised - more prone to surprise
            2,  # happy - more prone to happiness
            1.0,  # pleased
            1.0,  # fearful
            5,  # angry - more prone to anger
            2,  # grieved - more prone to sadness
            2,  # sad - more prone to sadness
            0.8,  # disgusted
            0.8,  # depressed
            0.8,  # tired
            1.0,  # calm
            1.0,  # relieved
        ]
        self.emotional_agent = EmotionalAgent(personality_weights=self.personality_weights)
        self.emotion_extractor = EmotionExtractor(role="Harry Potter")
        self.client = llmservice.client(service=service_type)
        
    def _generate_harry_prompt(self, user_input, current_emotion, next_emotion, history=[]):
        base_prompt = f"""You are Harry Potter, the famous wizard from the Harry Potter series. 
                    Currently you are feeling {next_emotion}. 
                    Respond to the following message while maintaining your character and emotional state.
                    Remember:
                    - You are brave and willing to stand up for what's right
                    - You care deeply about your friends
                    - You have experienced both great joy and deep loss
                    - You have a strong sense of justice
                    - You sometimes act impulsively
                    - Your response should reflect the change in your emotional state from {current_emotion} to {next_emotion}.

                    History chat: {history}
                    User current message: {user_input}

                    Respond as Harry Potter:"""
        return base_prompt


    def emotion_chat(self, user_input, history=[]):
        # Obtain current emotional state.
        current_state_idx = self.emotional_agent._find_nearest_state_index(self.emotional_agent.current_emotion)
        current_emotion = self.emotional_agent.emotion_states[current_state_idx].name
        
        # Analyze the reply and update the emotional state
        new_emotion = self.emotion_extractor.extract_emotion(user_input, model="doubao")
        new_state_idx = next(i for i, state in enumerate(self.emotional_agent.emotion_states) 
                           if state.name == new_emotion)
        
        # Calculating emotional changes
        current_vector = self.emotional_agent.emotion_states[current_state_idx].vector
        new_vector = self.emotional_agent.emotion_states[new_state_idx].vector
        delta_E = new_vector - current_vector
        
        # Update emotional states
        self.emotional_agent.update_transition_matrix(delta_E)
        next_state_idx = self.emotional_agent.get_next_emotion()
        self.emotional_agent.current_emotion = self.emotional_agent.emotion_states[next_state_idx].vector
        
        prompt = self._generate_harry_prompt(user_input, current_emotion, self.emotional_agent.emotion_states[next_state_idx].name)
        response = self.client.chat.completions.create(
            model= os.environ.get("ENDPOINT_ID"),
            messages=[{"role": "user", "content": prompt}]
        )
        harry_response = response.choices[0].message.content

        # Build response
        steps = [
            Step(
                explanation=f"Harry's current emotion: {current_emotion}",
                output="Processing message..."
            ),
            Step(
                explanation=f"Detected emotion from response: {new_emotion}",
                output=f"Updated emotional state: {self.emotional_agent.emotion_states[next_state_idx].name}"
            ),
            Step(
                explanation=f"Generated response as Harry Potter",
                output=harry_response
            ),
        ]
        
        return EmoResponse(
            steps=steps,
            emotion=self.emotional_agent.emotion_states[next_state_idx].name
        )
