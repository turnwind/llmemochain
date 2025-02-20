import numpy as np
from pydantic import BaseModel
from scipy.stats import multivariate_normal
from . import llmservice
import json
import os

class Step(BaseModel):
    explanation: str
    output: str

class EmoResponse(BaseModel):
    steps: list[Step]
    emotion: str

class EmotionState:
    def __init__(self, name, valence, arousal):
        self.name = name
        self.vector = np.array([valence, arousal])
        
class EmotionalAgent:
    def __init__(self, current_emotion = None, personality_weights=None, alpha=0.3):
        # Emotional State Space Configuration
        self.emotion_states = [
            EmotionState("surprised", 0.383, 0.924),
            EmotionState("happy", 0.707, 0.707),
            EmotionState("pleased", 0.924, 0.383),
            EmotionState("fearful", -0.383, 0.924),
            EmotionState("angry", -0.707, 0.707),
            EmotionState("grieved", -0.924, 0.383),
            EmotionState("sad", -0.924, -0.383),
            EmotionState("disgusted", -0.707, -0.707),
            EmotionState("depressed", -0.383, -0.924),
            EmotionState("tired", 0.383, -0.924),
            EmotionState("calm", 0.707, -0.707),
            EmotionState("relieved", 0.924, -0.383)
        ]
        
        # Dynamic Parameters Initialization
        self.current_emotion = current_emotion or [0,0] # Current Emotional Vector
        self.alpha = alpha  # Adjustment Term Scaling Factor
        
        # Weight Matrix Configuration
        self.W = np.array([s.vector for s in self.emotion_states])

        # Personality Weight Configuration
        self.personality_weights = personality_weights or np.ones(len(self.emotion_states))

        # Transition Matrix Initialization
        self.T0 = self._init_transition_matrix()
        self.T = self.T0.copy()
        
    def _init_transition_matrix(self):
        """Initialize the transition matrix based on the emotional space distances"""
        n = len(self.emotion_states)
        T = np.zeros((n, n))
        
        for i in range(n):
            distances = [np.linalg.norm(self.emotion_states[i].vector - s.vector) 
                        for s in self.emotion_states]
            similarities = 1 / (1 + np.array(distances))
            T[i] = similarities / similarities.sum()
            
        return T
    
    def _find_nearest_state_index(self, vector):
        """Map continuous vector to the nearest discrete state"""
        distances = [np.linalg.norm(vector - s.vector) 
                    for s in self.emotion_states]
        return np.argmin(distances)
    
    def update_transition_matrix(self, delta_E):
        """
        Update the transition matrix based on the emotional state changes
        :param delta_E: Expected emotion change μ (Δv, Δa)
        """
        k = self._find_nearest_state_index(self.current_emotion)
        
        # Calculate the adjustment term A = α * μ · W 
        A = self.alpha * np.dot(delta_E, self.W.T)
        # Update transition probabilities and apply personality weights.
        self.T[k] = (self.T0[k] + A) * self.personality_weights
        self.T[k] = np.where(self.T[k] < 0, 0, self.T[k])
        self.T[k] /= self.T[k].sum()  # Normalize to ensure probabilities sum to 1.
        
    def get_next_emotion(self, strategy='max', threshold=0.1):
        """
        Calculate the next emotional state
        :param strategy: Selection strategy (weighted/max/sample)
        :param threshold: Sampling threshold
        """
        k = self._find_nearest_state_index(self.current_emotion)
        probs = self.T[k]
        
        print("emotion probability:", {s.name: np.round(p, 2).item() for s, p in zip(self.emotion_states, probs)})

        # 策略选择
        if strategy == 'weighted':
            new_vector = sum(p*s.vector for p,s in zip(probs, self.emotion_states))
            return self._find_nearest_state_index(new_vector)
            
        elif strategy == 'max':
            return np.argmax(probs)
            
        elif strategy == 'sample':
            filtered_probs = np.where(probs > threshold, probs, 0)
            if filtered_probs.sum() == 0:
                filtered_probs = probs
            filtered_probs /= filtered_probs.sum()
            return np.random.choice(len(probs), p=filtered_probs)
        
    def process_input(self, input_mu, input_sigma=None):
        """
        Process external input
        :param input_mu: Expected emotion change μ
        :param input_sigma: Covariance matrix Σ (optional)
        """
        # Generate actual emotion change
        if input_sigma is not None:
            delta_E = multivariate_normal.rvs(mean=input_mu, cov=input_sigma)
        else:
            delta_E = input_mu
        
        # Update transition matrix
        self.update_transition_matrix(delta_E)
        
    def generate_response(self, context, strategy='weighted'):
        next_idx = self.get_next_emotion(strategy)
        next_emotion = self.emotion_states[next_idx]
        
        prompt = f"""
        [Context] {context}
        [Emotion Transition] From {self.current_emotion} to {next_emotion.vector}
        [Style Guide] 
        - Use {'subtle' if strategy=='weighted' else 'explicit'} emotional expressions
        - Incorporate physiological descriptions when arousal > 0.5
        """）
        
        self.current_emotion = next_emotion
        return f"Generated response for: {prompt}"
        
class EmotionExtractor:
    def __init__(self, role=None, service="doubao"):
        self.service = service
        self.client = llmservice.client(service=service)
        self.role = role
        self.states = [
            'surprised', 'happy', 'pleased', 'fearful', 'angry', 'grieved',
            'sad', 'disgusted', 'depressed', 'tired', 'calm', 'relieved'
        ]

        if self.role is None:
            self.pt = "What emotions do you feel when you hear the following conversation? Output the most likely emotion in {}.".format(self.states)
        else:
            self.pt = "You need to play the role of {}, what emotions will you feel when you hear the following conversation? Output the most likely emotion in {}.".format(self.role,self.states)
        
    
    def extract_emotion(self, says, pt=None, model="gpt-4o-mini"):
        self.pt = pt or self.pt

        if self.service == "deepseek":
            system_prompt = self.pt+ """
                            Please parse "emotion" and output them in JSON format. 
                            EXAMPLE JSON OUTPUT:
                            {
                                "emotion": "happy"
                            }
                            """
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": str(says)}]

            response = self.client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            response_format={'type': 'json_object'}
                            )

            try:
                emo = json.loads(response.choices[0].message.content)["emotion"]
            except:
                print("failed, try again")
                return self.extract_emotion(says, pt,model)

            if emo in self.states:
                return emo
            else:
                return self.extract_emotion(says, pt,model)
            
        elif self.service == "openai":
            chat_completion = self.client.beta.chat.completions.parse(
                messages=[
                    {
                        "role": "system",
                        "content": pt
                    },
                    {
                        "role": "user",
                        "content": str(says),
                    }
                ],
                model= model,
                response_format=EmoResponse,
            )
            
            message = chat_completion.choices[0].message
            if message.parsed and message.parsed.emotion.lower() in self.states:
                return message.parsed.emotion.lower()
            else:
                return self.extract_emotion(says, pt,model)
        else:
            system_prompt = self.pt+ """
                            Please parse "emotion" and output them in JSON format. 
                            EXAMPLE JSON OUTPUT:
                            {
                                "emotion": "happy"
                            }
                            """
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": str(says)}]

            response = self.client.chat.completions.create(
                model= os.environ.get("ENDPOINT_ID"),
                messages= messages,
                response_format={'type': 'json_object'}
            )
            try:
                emo = json.loads(response.choices[0].message.content)["emotion"]
            except:
                print("failed, try again")
                return self.extract_emotion(says, pt,model)

            if emo in self.states:
                return emo
            else:
                return self.extract_emotion(says, pt,model)