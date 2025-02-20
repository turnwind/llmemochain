import json
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.emotion import EmotionalAgent
from model.trainer import PersonalityWeightOptimizer
from model.harry_potter_llm import HarryPotterLLM

emotion_states = {
            "neutral": [0, 0],
            "surprised": [0.383, 0.924],
            "happy": [0.707, 0.707],
            "pleased": [0.924, 0.383],
            "fearful": [-0.383, 0.924],
            "angry": [-0.707, 0.707],
            "grieved": [-0.924, 0.383],
            "sad": [-0.924, -0.383],
            "disgusted": [-0.707, -0.707],
            "depressed": [-0.383, -0.924],
            "tired": [0.383, -0.924],
            "calm": [0.707, -0.707],
            "relieved": [0.924, -0.383]
        }
        

def load_training_data(data_path):
    """Load and preprocess training data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def calculate_emotion_distance(emotion1, emotion2):
    """Calculate Euclidean distance between two emotion vectors."""
    return np.sqrt(sum((e1 - e2) ** 2 for e1, e2 in zip(emotion1, emotion2)))

def main():
    # Initialize paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "data_en.json"
    
    # Load training data
    training_data = load_training_data(data_path)
    
    # Initialize Harry Potter agent
    harry = HarryPotterLLM()
    
    # Initialize optimizer
    optimizer = PersonalityWeightOptimizer(
        epsilon=0.05,
        alpha=0.1,
        beta=0.9,
        momentum=None
    )
    
    # Training loop
    print("Starting training for Harry Potter...")
    print("Initial personality weights:", harry.personality_weights)
    
    for episode in training_data:
        # Skip if not Harry's dialogue
        if episode.get("role") != "Harry":
            continue
        
        messages = episode['messages']
        for i in range(len(messages)):
            if messages[i]["role"] == "Harry" and i > 0:
                response = harry.emotion_chat(messages[i-1]["message"], history=messages[:i-1])
                pred_emotion = response.emotion
                true_emotion = messages[i]["emotion"]
                # Calculate emotion distance
                emotion_dist = calculate_emotion_distance(emotion_states[pred_emotion], emotion_states[true_emotion])
                
                # Get personality idx
                for idx, em in enumerate(emotion_states):
                    if em == true_emotion:
                        true_personality_idx = idx
                        break
                
                # Optimize weights
                new_weights = optimizer.optimize_weights(
                    current_weights=harry.personality_weights,
                    true_emotion_idx=true_personality_idx,
                    emotion_distance=emotion_dist
                )
            
        # Update agent's weights
        harry.personality_weights = new_weights
        print(f"Updated weights: {new_weights}")

    print("\nTraining completed!")
    print("Final personality weights:", harry.personality_weights)

if __name__ == "__main__":
    # Train 3 times
    for i in range(3):
        main()
