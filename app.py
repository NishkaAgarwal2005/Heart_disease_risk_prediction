import streamlit as st
import requests
import json
import numpy as np
import joblib
import pandas as pd
import os
import speech_recognition as sr
from streamlit_login_auth_ui.widgets import __login__
import warnings
import io
from fpdf import FPDF
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# 🛠️ Load Model
@st.cache_resource
def load_model():
    return joblib.load('model_pipeline.joblib')

# Initialize Session State with a clear structure
# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "health_tips" not in st.session_state:
    st.session_state["health_tips"] = ""
if "health_trends" not in st.session_state:
    st.session_state["health_trends"] = []
if "voice_input" not in st.session_state:
    st.session_state["voice_input"] = ""
if "selected_page" not in st.session_state:
    st.session_state["selected_page"] = "Prediction"


# BMI Calculation Function (Unchanged)
def calculate_bmi(weight, height):
    height_m = height / 100
    return weight / (height_m ** 2)

# 🔹 Chatbot for Health Queries (Using Gemini API)
def chatbot_response(user_query):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    api_key = st.secrets["api"]["gemini_key"]  # Secure API Key Handling

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": user_query}]}]}

    response = requests.post(f"{api_url}?key={api_key}", json=payload, headers=headers)

    if response.status_code == 200:
        try:
            return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        except KeyError:
            return "Sorry, I couldn't process your query. Try again!"
    else:
        return f"API Error: {response.status_code} - {response.text}"
    
# 🔹 Function to Generate Health Tips (Using Gemini API)
def generate_health_tips(user_data):
    prompt = f"""
    You are a health expert providing cardiovascular health tips. Address the user directly as 'you' instead of 'he' or 'she'. 
    Based on the following health data, give 3 actionable and personalized health tips in simple language:
    if it is predicted to be a high-risk patient then also suggest some exercises or yoga poses to reduce the risk of heart disease.
    - Age: {user_data['age'] // 365} years
    - Gender: {user_data['gender']}
    - Height: {user_data['height']} cm
    - Weight: {user_data['weight']} kg
    - BMI: {calculate_bmi(user_data['weight'], user_data['height']):.2f}
    - Systolic BP: {user_data['ap_hi']} mmHg
    - Diastolic BP: {user_data['ap_lo']} mmHg
    - Cholesterol Level: {user_data['cholesterol']}
    - Glucose Level: {user_data['gluc']}
    - Smokes: {user_data['smoke']}
    - Drinks Alcohol: {user_data['alco']}
    - Physically Active: {user_data['active']}
    
    Your response should start directly with the first tip, without any introductions like 'Here are three tips'.
    Just list the tips one after another, ensuring they are distinct, meaningful, and concise.
    """

    return chatbot_response(prompt)  # Use Gemini AI for health tips

# 🎤 Function to Capture Voice Input
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand that."
    except sr.RequestError:
        return "Voice recognition error."

# 🔹 Function to Generate PDF Report
def generate_pdf(prediction, tips):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, "Cardio Disease Prediction Report", ln=True, align="C")
    pdf.ln(10)

    pdf.cell(200, 10, f"Prediction: {'High Risk' if prediction == 1 else 'Low Risk'}", ln=True)
    pdf.ln(5)

    pdf.cell(200, 10, "Health Tips:", ln=True)
    pdf.ln(5)

    for tip in tips:
        sanitized_tip = tip.encode('latin-1', 'replace').decode('latin-1')  # Replace unsupported characters
        pdf.multi_cell(0, 10, f"- {sanitized_tip}")

    # ✅ Save the PDF to an in-memory buffer correctly
    pdf_buffer = io.BytesIO()
    pdf_output = pdf.output(dest="S").encode("latin1")  # Get PDF as bytes
    pdf_buffer.write(pdf_output)
    pdf_buffer.seek(0)  # Move to the beginning of the buffer

    return pdf_buffer

# 🔹 User Input Function
def user_input_features():
    birth_date = st.date_input("Birth Date", value=datetime(1990, 1, 1),
                               min_value=datetime.now().date() - timedelta(days=100*365),
                               max_value=datetime.now().date() - timedelta(days=18*365))
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)
    ap_hi = st.number_input("Systolic BP", min_value=50, max_value=250, value=120)
    ap_lo = st.number_input("Diastolic BP", min_value=30, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])
    gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
    smoke = st.selectbox("Smoking", ["Yes", "No"])
    alco = st.selectbox("Alcohol", ["Yes", "No"])
    active = st.selectbox("Physical Activity", ["Yes", "No"])

    data = {
        'age': (datetime.now().date() - birth_date).days,
        'gender': gender,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active
    }
    
    return pd.DataFrame(data, index=[0])

__login__obj = __login__(auth_token = "dk_prod_XHG9DC6V4EMCB2J8X6GJA01AFJMS", 
                    company_name = "Shims",
                    width = 200, height = 250, 
                    logout_button_name = 'Logout', hide_menu_bool = False, 
                    hide_footer_bool = False, 
                    lottie_url = 'https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json')

LOGGED_IN = __login__obj.build_login_ui()

# Main App Logic
if LOGGED_IN == True:
    # Sidebar Navigation
    st.sidebar.title("🔍 Menu")
    selected_page = st.sidebar.radio(
        "Go to", ["Prediction", "Chatbot", "Health Trends", "Advanced Functionalities"]
    )


        # Display the Selected Page
    if selected_page == "Prediction":
        st.title("Cardio Disease Prediction")
        st.image("heart-disease.jpg", use_column_width=True)
        input_df = user_input_features()

        if st.button("Predict"):
            model = load_model()
            prediction = model.predict(input_df)[0]
            st.session_state["prediction"] = prediction

            health_tips = generate_health_tips(input_df.iloc[0].to_dict())
            st.session_state["health_tips"] = health_tips  # Store in session 

            if prediction == 1:
                st.error("Prediction: High Risk")
            else:
                st.success("Prediction: Low Risk")
            for tip in health_tips.split("\n"):
                st.write(f"- {tip}")

            st.session_state["health_trends"].append(input_df.iloc[0].to_dict())

                # ✅ Generate PDF Only if Prediction Exists
            if st.session_state["prediction"] is not None:
                pdf_buffer = generate_pdf(st.session_state["prediction"], 
                st.session_state["health_tips"].split("\n"))

                st.download_button(
                    label="Download Report",
                    data=pdf_buffer,
                    file_name="health_report.pdf",
                    mime="application/pdf"
                )

        # (Previous code remains the same, only modifying the Chatbot section)

    elif selected_page == "Chatbot":
        st.title("💬 Health Chatbot")

        # Option to input via text or voice
        input_method = st.radio("Choose Input Method", ["Text", "Voice"])

        if input_method == "Text":
            chat_input = st.text_input("Type your health question:")
            input_to_use = chat_input
            voice_disabled = True #disable voice option
        else:
            st.info("Click '🎤 Speak Question' to start voice input")
            input_to_use = st.session_state.get("voice_input", "")
            voice_disabled = False #enable voice option

        # Voice input button
        if st.button("🎤 Speak Question", disabled = voice_disabled):
            try:
                voice_input = recognize_speech()
                st.session_state["voice_input"] = voice_input
                input_to_use = voice_input
                st.write(f"Recognized: {voice_input}")
            except Exception as e:
                st.error(f"Voice recognition error: {e}")

        # Button to get response
        if st.button("Get Answer"):
            if input_to_use:
                try:
                    response = chatbot_response(input_to_use)
                    st.write("Response:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error getting response: {e}")
            else:
                st.warning("Please enter a question first.")


    elif selected_page == "Health Trends":
        st.title("📊 Health Trends")
        st.markdown("Track your health trends over time.")
    # 🔹 Function to Generate Dynamic Health Chart Analysis from Gemini API
        def generate_chart_analysis(ap_hi, ap_lo, weight):
            prompt = f"""
            You are a cardiovascular health expert.

            Analyze the following user's chart data, which includes three parameters:
            - ap_hi: Systolic blood pressure (dark blue line)
            - ap_lo: Diastolic blood pressure (light blue line)
            - weight: Body weight (red line)

            Based on the values provided below, give:
                1. What each line color represents
                2. What each parameter measures and its importance
                3. What changes you observe (increasing/decreasing/stable)
                4. What these changes imply for cardiovascular health
                5. A table summarizing variations and their health effects
                6. A summary conclusion and 4 specific health recommendations

                Use plain and easy-to-understand language.
                Speak directly to the user ("you").

                Values (latest to oldest):
                - ap_hi: {ap_hi}
                - ap_lo: {ap_lo}
                - weight: {weight}
            """
            return chatbot_response(prompt)


        if st.session_state["health_trends"]:
            df = pd.DataFrame(st.session_state["health_trends"])
            st.line_chart(df[['ap_hi', 'ap_lo', 'weight']])

            # Use recent 4 values (latest to oldest)
            ap_hi_list = df['ap_hi'].tolist()[-4:]
            ap_lo_list = df['ap_lo'].tolist()[-4:]
            weight_list = df['weight'].tolist()[-4:]

            if st.button("🔍 Analyze Chart"):
                with st.spinner("Analyzing your chart..."):
                    analysis = generate_chart_analysis(ap_hi_list, ap_lo_list, weight_list)

                st.subheader("📄Health Insights")
                st.markdown(analysis)
        else:
            st.warning("No health trends data available. Please add entries first.")


    elif selected_page == "Advanced Functionalities":
        st.title("🚀 Advanced Functionalities")
        st.info("Future features will be added here!")

# Run the main application
