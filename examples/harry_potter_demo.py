import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.harry_potter_llm import HarryPotterLLM

def main():
    # 初始化哈利波特 LLM
    harry = HarryPotterLLM()
    
    conversations = [
    "Hey, I was really looking forward to catching up today!",
    "I'm sorry—I got caught up at work and totally lost track of time.",
    "I have to admit, it hurts because I was excited to spend time together.",
    "I understand, and I promise to be more mindful next time."
]

    history = []
    for message in conversations:
        print("\nUser:", message)
        response = harry.emotion_chat(message, history=history)
        print("\nHarry's Response:")
        for step in response.steps:
            print(f"\n{step.explanation}:")
            print(step.output)
        print("\nFinal Emotion:", response.emotion)
        print("-" * 80)

        # 添加历史记录
        history.append({"user": message})
        history.append({"harry": response})

if __name__ == "__main__":
    main()
