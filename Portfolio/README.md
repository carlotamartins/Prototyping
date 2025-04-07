# Medical Prediction & Assistant Tool

This internal web application is designed for healthcare professionals to predict **readmission risk** for diabetic patients. Powered by AI and clinical data, the app helps doctors analyze patient profiles, generate reports, and interact with an intelligent assistant for data-driven guidance.

---

## Features

- View raw patient medical data
- Predict readmission risk using a trained ML model
- Download personalized summary PDFs
- Chat with an AI assistant (powered by Cohere) for clinical recommendations
- Professionally styled UI with background image and sidebar navigation

---

## 🗂Project Structure

| File | Description |
|------|-------------|
| `diabetes_st_final.py` | Main Streamlit app file that powers the UI and logic |
| `database.csv` | Sample dataset containing patient info |
| `ml_model.pkl` | Pre-trained machine learning model for predicting readmission |
| `style.css` | Custom styles for app appearance (fonts, layout, colors) |
| `background.png` | Background image used in the UI |
| `cohere.key` | API key file for accessing Cohere's AI assistant (not shared in public repos) |
| `model.ipynb` | Jupyter notebook used to train and evaluate the prediction model |
| `requirements.txt` | List of Python packages required to run the app |

---

## How to Run Locally

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or .\\venv\\Scripts\\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Cohere key (replace this line in the script with os.getenv if needed)
echo "your-cohere-key" > cohere.key

# 4. Run the app
streamlit run diabetes_st_final.py
