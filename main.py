
!pip install -q -U google-genai gradio scikit-learn pandas

import os, pandas as pd, gradio as gr
from google import genai
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


import os
from google import genai

from google.colab import userdata
API_KEY = userdata.get('GOOGLE_API_KEY')

os.environ["GOOGLE_API_KEY"] = API_KEY
client = genai.Client(api_key=API_KEY)

from google.colab import files
uploaded = files.upload()

df = pd.read_csv(next(iter(uploaded)))
df["Symptoms"] = df[df.columns[1:]].apply(lambda x: ', '.join(x.dropna()), axis=1)
le = LabelEncoder()
df["disease_label"] = le.fit_transform(df["Disease"])

X_train, X_test, y_train, y_test = train_test_split(
    df["Symptoms"], df["disease_label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

def health_coach(symptoms):

    sympt_vec = vectorizer.transform([symptoms])
    pred_label = model.predict(sympt_vec)[0]
    disease = le.inverse_transform([pred_label])[0]


    prompt = f"""
    You are a personal health coach. Given these symptoms: {symptoms}
    The predicted disease is: {disease}.
    Please provide advice on next steps, home care, and when to consult a professional.
    """
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return f"**Predicted Disease:** {disease}\n\n**Gemini Advice:**\n{response.text}"

iface = gr.Interface(
    fn=health_coach,
    inputs=gr.Textbox(lines=3, placeholder="e.g. fever, fatigue, dizziness"),
    outputs="markdown",
    title="AI-Powered Personal Health Coach",
    description="Enter symptoms to get a predicted disease and advice."
)
iface.launch(share=True)
