from transformers import pipeline
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
import requests

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Personal Fitness Tracker", layout="centered")
st.title("🏋️‍♂️ Personal Fitness Tracker")
st.markdown("⚠️ **Disclaimer:** The AI's health suggestions are for informational purposes only and should NOT be considered medical advice. Always consult with qualified healthcare professionals for any health concerns.")

# --- User Name Input ---
person_name = st.sidebar.text_input("Your Name", placeholder="Enter your name")
if person_name:
    st.sidebar.markdown(f"Welcome, **{person_name}!** 👋")
else:
    st.sidebar.markdown("Welcome! 👋")

# --- Load Local AI Model (Fallback) ---
@st.cache_resource
def load_local_ai():
    try:
        return pipeline("text2text-generation", model="google/flan-t5-small")
    except Exception as e:
        st.error(f"Error loading local AI model: {e}")
        return None

llm_local = load_local_ai()

def ai_health_response_local(query):
    if llm_local:
        prompt = f"Give a general health suggestion for {person_name if person_name else 'the user'}: {query}"
        try:
            response = llm_local(prompt, max_new_tokens=80)[0]['generated_text']
            return response
        except Exception as e:
            st.error(f"Error generating response with local AI: {e}")
            return "Error generating local AI response."
    else:
        return "Local AI model not loaded."

# --- Hugging Face Inference API Integration ---
HF_API_TOKEN = ""  # You can try your token here again, but we'll prioritize local if it fails
HF_MODEL_NAME = "google/flan-t5-large"  # A publicly accessible model

@st.cache_resource
def get_huggingface_headers():
    if HF_API_TOKEN:
        return {"Authorization": f"Bearer {HF_API_TOKEN}"}
    return None

hf_headers = get_huggingface_headers()
hf_api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}"

def ai_health_response_realtime(query):
    if hf_headers:
        payload = {"inputs": f"Give a concise and helpful health suggestion for {person_name if person_name else 'the user'}: {query}"}
        try:
            response = requests.post(hf_api_url, headers=hf_headers, json=payload)
            response.raise_for_status()
            return response.json()[0]['generated_text']
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching AI response from Hugging Face: {e}")
            return f"Error fetching AI response from Hugging Face."
        except (KeyError, IndexError, ValueError) as e:
            st.error(f"Error processing AI response from Hugging Face: {e}")
            return f"Error processing AI response from Hugging Face."
    else:
        return "Hugging Face API token not provided."

# --- Ask Health AI Section ---
st.header("🤖 Ask Health AI")
query_ai = st.text_input("Ask a health question", placeholder="e.g. What are some tips for better sleep?")

use_realtime_ai = st.checkbox("Use Real-time AI (Hugging Face - might not work)", value=False)

if use_realtime_ai and not HF_API_TOKEN:
    st.warning("Hugging Face API token not found in the code. Using local AI.")

if st.button("Get AI Suggestion"):
    if query_ai.strip():
        if use_realtime_ai and HF_API_TOKEN:
            response = ai_health_response_realtime(query_ai)
            st.markdown(f"**{person_name if person_name else 'AI'}'s Suggestion:** {response}")
        elif llm_local:
            response = ai_health_response_local(query_ai)
            st.markdown(f"**{person_name if person_name else 'AI'}'s Suggestion:** {response}")
        else:
            response = "AI functionality is currently unavailable."
        st.markdown(f"**{person_name if person_name else 'AI'}'s Suggestion:** {response}")
    else:
        st.warning("Please enter a question.")

# --- Fitness Tracking and Health Knowledge Base (Rest of your code - unchanged) ---
# MET values for expanded activities
MET_VALUES = {
    "walking": 3.5, "running": 7.5, "cycling": 6.8, "swimming": 5.8,
    "yoga": 2.5, "weights": 4.0, "dancing": 5.0, "aerobics": 6.0,
    "hiking": 6.0, "jumping rope": 10.0, "rowing": 7.0, "riding": 4.5,
    "skating": 7.0, "football": 8.0, "basketball": 6.5
}

HEALTH_KNOWLEDGE_BASE = {
    "diabetes": {
        "description": "A chronic condition that affects how your body turns food into energy.",
        "tips": [
            "Follow a low-carb, high-fiber diet.",
            "Exercise regularly (e.g., walking, cycling).",
            "Monitor blood sugar daily.",
            "Avoid sugary snacks and beverages."
        ]
    },
    "high blood pressure": {
        "description": "Also called hypertension; can lead to heart issues if untreated.",
        "tips": [
            "Reduce salt intake.",
            "Exercise daily for 30 minutes.",
            "Avoid smoking and alcohol.",
            "Eat potassium-rich foods like bananas."
        ]
    },
    "obesity": {
        "description": "A medical condition involving excess body fat.",
        "tips": [
            "Eat in a calorie deficit.",
            "Avoid processed food and sugars.",
            "Drink water before meals.",
            "Walk at least 10,000 steps daily."
        ]
    },
    "headache": {
        "description": "Pain in the head region, often due to stress, dehydration, or sleep issues.",
        "tips": [
            "Drink plenty of water.",
            "Practice stress-relieving activities like meditation.",
            "Avoid excessive screen time.",
            "Get 7-8 hours of sleep."
        ]
    },
    "fatigue": {
        "description": "Feeling overtired, with low energy and motivation.",
        "tips": [
            "Ensure consistent sleep schedule (7-8 hrs).",
            "Stay hydrated.",
            "Limit caffeine late in the day.",
            "Take short walks to refresh energy."
        ]
    }
}

@st.cache_data
def init_data():
    return pd.DataFrame(columns=["Date", "Activity", "Duration", "Calories", "Weight", "Height", "BMI"])

def parse_input(text):
    duration_match = re.search(r"(\d+)\s*(min|minutes?)", text.lower())
    if not duration_match:
        raise ValueError("Please include duration like '30 minutes' in your input.")

    duration = int(duration_match.group(1))
    activity = next((a for a in MET_VALUES if a in text.lower()), None)
    if not activity:
        raise ValueError("Couldn't detect a known activity. Try including words like running, cycling, dancing, etc.")

    return activity, duration

def estimate_calories(activity, duration, weight):
    met = MET_VALUES.get(activity, 3.5)
    return round(met * weight * (duration / 60), 2)

def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return round(weight / (height_m ** 2), 2)

def give_food_recommendation(goal):
    food = {
        "gain": ["Peanut butter toast", "Chicken breast + rice", "Oats with banana", "Eggs", "Milkshake"],
        "maintain": ["Veg sandwich", "Tofu stir-fry", "Paneer salad", "Fruit bowl", "Greek yogurt"],
        "lose": ["Boiled eggs", "Soup", "Veg wrap", "Cucumber salad", "Green tea"]
    }
    return food.get(goal, [])

def determine_goal(bmi):
    if bmi < 18.5:
        return "gain"
    elif bmi <= 24.9:
        return "maintain"
    else:
        return "lose"

def fetch_health_suggestions(query):
    query = query.lower().strip()
    for key in HEALTH_KNOWLEDGE_BASE:
        if key in query:
            result = HEALTH_KNOWLEDGE_BASE[key]
            tips = "\n".join([f"- {tip}" for tip in result["tips"]])
            return f"**{result['description']}**\n\n**Tips:**\n{tips}"
    return "Sorry, we couldn’t find suggestions for that issue. Try general terms like 'diabetes', 'headache', or 'fatigue'."

# --- Sidebar for Personal Info ---
st.sidebar.title("👤 Personal Info")
weight = st.sidebar.number_input("Weight (kg)", 40, 200, 70)
height = st.sidebar.number_input("Height (cm)", 140, 220, 170)
bmi = calculate_bmi(weight, height)
st.sidebar.metric("BMI", bmi)
goal = determine_goal(bmi)
st.sidebar.write("Goal:", goal.upper())

# --- Workout Logging Section ---
st.header("📝 Log Your Workout")
text_input = st.text_input("Describe your workout", placeholder="e.g. I did 30 minutes of cycling")

if 'df' not in st.session_state:
    st.session_state.df = init_data()

if st.button("Add Workout Entry"):
    if text_input:
        try:
            activity, duration = parse_input(text_input)
            calories = estimate_calories(activity, duration, weight)
            new_entry = pd.DataFrame([{
                "Date": datetime.today().strftime('%Y-%m-%d'),
                "Activity": activity.capitalize(),
                "Duration": duration,
                "Calories": calories,
                "Weight": weight,
                "Height": height,
                "BMI": bmi
            }])
            st.session_state.df = pd.concat([st.session_state.df, new_entry], ignore_index=True)
            st.success(f"Added {activity} for {duration} min - ~{calories} cal burned")
        except ValueError as e:
            st.error(str(e))

# --- Weekly Summary and Meal Suggestions ---
st.header("📊 Weekly Summary")
if not st.session_state.df.empty:
    df = st.session_state.df.copy()
    st.dataframe(df.tail(7))
    st.bar_chart(df.groupby("Activity")["Calories"].sum())

    st.subheader("🍱 Meal Suggestions")
    st.markdown("Suggested foods based on your goal:")
    for item in give_food_recommendation(goal):
        st.markdown(f"- {item}")
else:
    st.info("No workouts logged yet.")

# --- Health Advice from Internal Knowledge Base ---
st.header("🩺 Got a Health Issue? (Internal)")
query_kb = st.text_input("Type a symptom or health question", placeholder="e.g. diabetes, headache, fatigue")
if st.button("Get Internal Suggestion"):
    if query_kb.strip():
        suggestion = fetch_health_suggestions(query_kb)
        st.markdown(suggestion)
    else:
        st.warning("Please type something!")
