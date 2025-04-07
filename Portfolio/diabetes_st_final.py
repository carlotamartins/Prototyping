import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import cohere
import os
import speech_recognition as sr
import plotly.express as px
import base64
from streamlit_option_menu import option_menu


# Load trained model and transformer
model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_model.pkl"))

def apply_custom_style():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #dbeeff; border-radius: 10px;'>
        <h1 style='margin-bottom: 0.2rem;'>Medical Prediction & Assistant Tool</h1>
        <p style='color: #555;'>Powered by AI • Designed for Healthcare Professionals</p>
    </div>
    """, unsafe_allow_html=True)

apply_custom_style()


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('background.png')


def load_data():
    df = pd.read_csv('database.csv')
    df = df[['gender', 'age', 'time_in_hospital', 'num_lab_procedures', 'insulin', 'diabetesMed', 'readmitted']]
    return df

def display_sidebar():
    st.sidebar.title("Medical Prediction App")
    st.sidebar.info("This app helps predict patient readmission risk based on medical data.")
    with st.sidebar:
        section = option_menu(
            menu_title=None,
            options=[
                "Patient Data",
                "Customizable Predictions",
                "Download Patient Summary",
                "AI Chat Assistant"
            ],
            icons=["table", "activity", "file-earmark-arrow-down", "robot"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#e3f2fd"},
                "icon": {"color": "#1a237e", "font-size": "20px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "10px",
                    "padding": "12px",
                    "border-radius": "10px",
                    "background-color": "#f8fbff",
                },
                "nav-link-selected": {
                    "background-color": "#64b5f6",
                    "color": "white",
                },
            }
        )
    return section

def display_raw_data(df):
    st.subheader("Raw Data")
    st.dataframe(df)

def preprocess_input(gender, insulin, diabetesMed, age, time_in_hospital, num_lab_procedures):
    processed_data = pd.DataFrame([[
        time_in_hospital,
        num_lab_procedures,
        age,
        insulin,
        gender,
        diabetesMed,
    ]], columns=['time_in_hospital', 'num_lab_procedures', 'age', 'insulin', 'gender', 'diabetesMed'])
    return processed_data

def make_prediction(user_input):
    prediction = model.predict(user_input)
    return prediction[0]


def generate_pdf(patient_info, prediction):
    pdf_output = BytesIO()
    pdf = canvas.Canvas(pdf_output, pagesize=letter)
    
    # Title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(100, 770, "Patient Report")

    y_position = 740

    pdf.setFont("Helvetica", 12)
    patient_info_labels = ["Gender", "Insulin", "Diabetes Medication", "Age", "Time in Hospital", "Number of Lab Procedures"]
    for label, value in zip(patient_info_labels, patient_info):
        pdf.drawString(100, y_position, f"{label}: {value}")
        y_position -= 20

    # Prediction and Recommendations (just once)
    y_position -= 20 
    pdf.setFont("Helvetica-Bold", 12)
    prediction_text = "High risk of readmission" if prediction == 1 else "Low risk of readmission"
    pdf.drawString(100, y_position, f"Prediction: {prediction_text}")
    y_position -= 20

    pdf.drawString(100, y_position, "Recommendations:")
    y_position -= 20

    pdf.setFont("Helvetica", 12)
    recommendations = [
        "1. Schedule an appointment with your healthcare provider as soon as possible.",
        "2. Monitor your blood sugar levels regularly.",
        "3. Follow a balanced diet rich in fruits, vegetables, and whole grains.",
        "4. Take prescribed medications as directed by your healthcare provider.",
        "5. Engage in regular physical activity, such as walking or swimming.",
        "6. Attend diabetes education classes to better manage your condition.",
        "7. Report any unusual symptoms to your healthcare provider immediately."
    ] if prediction == 1 else [
        "1. Maintain a balanced diet and healthy lifestyle.",
        "2. Continue taking prescribed medications as directed.",
        "3. Keep up with regular physical activity.",
        "4. Schedule regular check-ups with your healthcare provider.",
        "5. Monitor your blood sugar levels regularly.",
        "6. Stay informed about diabetes management through educational resources."
    ]

    for recommendation in recommendations:
        pdf.drawString(120, y_position, recommendation)
        y_position -= 20

    pdf.showPage()
    pdf.save()
    pdf_output.seek(0)
    return pdf_output.getvalue()


with open('cohere.key', 'r') as file:
    cohere_api_key = file.read().strip()

co = cohere.Client(cohere_api_key)

def capture_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎧 Listening for your question...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.success(f"Captured: {text}")
            return text
        except sr.UnknownValueError:
            st.warning("Could not understand the audio.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")


def extract_plot_vars(user_input, df_columns):
    user_input = user_input.lower()
    x_col = y_col = None
    for col in df_columns:
        if col.lower() in user_input:
            if not x_col:
                x_col = col
            elif not y_col and col != x_col:
                y_col = col
    return x_col, y_col



def smart_assistant(df, patient_info, prediction):
    st.subheader("🧠 Ask Dr. Fridolin!")

    prompt = f"""
You are a clinical assistant for internal medicine doctors. The following is patient information and a predicted risk of readmission. 
Based on this, suggest which clinical features require further attention and what actionable steps the physician could take.

Patient Info:
- Gender: {patient_info[0]}
- Insulin: {patient_info[1]}
- Diabetes Medication: {patient_info[2]}
- Age: {patient_info[3]}
- Time in Hospital: {patient_info[4]}
- Number of Lab Procedures: {patient_info[5]}
- Predicted Risk: {'High' if prediction == 1 else 'Low'}

Recommendations should be:
- Clinically grounded
- Clear and to the point
- In bullet point format
- Diabetes related
"""

    with st.spinner("Analyzing patient case..."):
        try:
            response = co.generate(prompt=prompt, model="command-light", temperature=0.3)
            response_text = response.generations[0].text.strip()
            st.markdown("**🔍 Clinical Focus Areas:**")
            for line in response_text.split("\n"):
                if line.strip():
                    st.markdown(f"<p>✅ {line.strip()}</p>", unsafe_allow_html=True)

            if any("visualize" in line.lower() for line in response_text.split("\n")):
                if st.button("Show Suggested Chart"):
                    st.info("Trend in age and lab procedures...")
                    age_lab = df.groupby("age")["num_lab_procedures"].mean().reset_index()
                    fig = px.line(age_lab, x="age", y="num_lab_procedures", title="Average Lab Procedures by Age")
                    st.plotly_chart(fig)

            if st.button("Show Clinical Summary"):
                summary_prompt = f"""
    Summarize this patient's situation in 3 concise, clinical sentences for a physician reviewing the case:
    {prompt}
    """
                summary_response = co.generate(prompt=summary_prompt, model="command-light", temperature=0.3)
                st.markdown(f"**📄 Summary:** {summary_response.generations[0].text.strip()}")

        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("**Ask a Question** (via Text or Voice)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history[-6:]:
        st.markdown(f"**{role}:** {msg}")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_input("Type a question:", key="unified_input")
    with col2:
        if st.button("🎤 Voice Input"):
            voice_result = capture_voice_input()
            if voice_result:
                st.session_state.chat_history.append(("You (via voice)", voice_result))
                user_input = voice_result

    if user_input:
        try:
            conversation = "\n".join([f"{role}: {msg}" for role, msg in st.session_state.chat_history[-6:]])
            chat_prompt = f"""
{conversation}
Doctor's Question: {user_input}
Assistant:
"""
            response = co.generate(prompt=chat_prompt, model="command-light", temperature=0.3)
            reply = response.generations[0].text.strip()
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", reply))
            st.markdown(f"**Bot:** {reply}")

            columns = list(df.columns)

            # Always try to extract columns to plot
            x_col, y_col = extract_plot_vars(user_input, columns)

            if x_col and y_col:
                st.info(f"📊 Generating plot for: {x_col} vs {y_col}")
                fig = px.scatter(df, x=x_col, y=y_col, color="readmitted", title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig)
            else:
                # If not found, try fallback using intent classification
                intent_prompt = f"Classify the intent of this question: '{user_input}'"
                intent_response = co.generate(prompt=intent_prompt, model="command-light", temperature=0.3)
                intent = intent_response.generations[0].text.strip().lower()

                if intent in ["plot", "compare", "trend"]:
                    st.warning("Bot detected intent to plot but couldn't extract valid features. Try rephrasing.")


        except Exception as e:
            st.error(f"Error: {e}")


def main():
    section = display_sidebar()
    df = load_data()
    if section == "Patient Data":
        display_raw_data(df)
    elif section == "Customizable Predictions":
        st.subheader("Make a Prediction")
        gender = st.selectbox("Gender", ['Female', 'Male', 'Unknown/Invalid'])
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        time_in_hospital = st.number_input("Time in Hospital", min_value=0, step=1)
        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, step=1)
        insulin = st.selectbox("Insulin", ['No', 'Up', 'Steady', 'Down'])
        diabetesMed = st.selectbox("Diabetes Medication", ['No', 'Yes'])
        if st.button("Predict"):
            user_input = preprocess_input(gender, insulin, diabetesMed, age, time_in_hospital, num_lab_procedures)
            prediction = make_prediction(user_input)
            if prediction is not None:
                st.success(f"Prediction: {'Readmitted' if prediction == 1 else 'Not Readmitted'}")
                st.session_state.prediction_result = prediction
                st.session_state.patient_info = [gender, insulin, diabetesMed, age, time_in_hospital, num_lab_procedures]
    elif section == "Download Patient Summary":
        st.subheader("Download Patient Report")
        if "prediction_result" in st.session_state:
            pdf = generate_pdf(st.session_state.patient_info, st.session_state.prediction_result)
            st.download_button("Download Report", data=pdf, file_name="patient_report.pdf", mime="application/pdf")
        else:
            st.info("Please run a prediction first.")
    elif section == "AI Chat Assistant":
        if "prediction_result" in st.session_state:
            smart_assistant(df, st.session_state.patient_info, st.session_state.prediction_result)
        else:
            st.info("Please run a prediction first.")



if __name__ == "__main__":
    main()
