# src/llm/groq_client.py

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def explain_emotion(emotion, confidence):
    prompt = f"""
    A facial emotion recognition model predicted:

    Emotion: {emotion}
    Confidence: {confidence:.2f}

    Give a short human-friendly explanation.
    Also if the user talks about something other than emotion
    just continute the conversation but under 40 tokens.
    """

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.6,
	    max_completion_tokens=256,
            reasoning_effort="low",
            include_reasoning=False
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Groq Error: {str(e)}"
