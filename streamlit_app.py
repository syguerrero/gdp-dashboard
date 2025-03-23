import streamlit as st
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title='Éxito Profesional',
    page_icon=':mortar_board:',
)

st.title("🎓 Dashboard de Éxito Profesional")
st.write("Sube un archivo CSV con datos de alumnos para predecir su probabilidad de éxito (salario alto y satisfacción profesional).")

data_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if data_file:
    df = pd.read_csv(data_file)
    original_df = df.copy()

    required_columns = ["Starting_Salary", "Career_Satisfaction", "Gender", "Field_of_Study", "Current_Job_Level", "Entrepreneurship"]
    if not all(col in df.columns for col in required_columns):
        st.error("El archivo CSV no contiene todas las columnas necesarias.")
    else:
        df["Exito"] = ((df["Starting_Salary"] >= 45000) & (df["Career_Satisfaction"] >= 7)).astype(int)

        X = df.drop(columns=["Exito", "Student_ID", "Career_Satisfaction", "Job_Offers"], errors='ignore')
        y = df["Exito"]

        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)

        st.subheader("Datos cargados")
        st.dataframe(original_df.head())

        st.subheader("Resultados de predicción")
        prob = clf.predict_proba(X)[:, 1]
        original_df["Probabilidad_Exito"] = (prob * 100).round(2)
        original_df["Es_Exitoso"] = clf.predict(X)

        st.dataframe(original_df[["Probabilidad_Exito", "Es_Exitoso"] + [col for col in original_df.columns if col not in ["Probabilidad_Exito", "Es_Exitoso"]]].head())

        csv = original_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar resultados con predicciones",
            data=csv,
            file_name="predicciones_exito.csv",
            mime="text/csv"
        )
