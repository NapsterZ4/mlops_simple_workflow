import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():
    return joblib.load(filename='model/logistic_regression_model.pkl')


model = load_model()

st.title('Predicción de Cáncer de Mama')
st.header('Ingrese los datos del paciente:')

inputs = {}
for feature in ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean',
                'compactness_mean', 'concavity_mean', 'concave points_mean',
                'symmetry_mean',
                'fractal_dimension_mean', 'radius_se', 'texture_se',
                'perimeter_se', 'area_se',
                'smoothness_se', 'compactness_se', 'concavity_se',
                'concave points_se',
                'symmetry_se', 'fractal_dimension_se', 'radius_worst',
                'texture_worst',
                'perimeter_worst', 'area_worst', 'smoothness_worst',
                'compactness_worst',
                'concavity_worst', 'concave points_worst', 'symmetry_worst',
                'fractal_dimension_worst']:
    inputs[feature] = st.number_input(f'Ingrese {feature}:', min_value=0.0,
                                      max_value=1000.0, value=0.0)

if st.sidebar.button('Realizar predicción'):
    df_new_data = pd.DataFrame(data=inputs, index=[0])

    prediction = model.predict(df_new_data)

    st.header('Resultado de la predicción:')
    if prediction[0] == 0:
        st.write('El resultado sugiere que el tumor es benigno.')
    else:
        st.write('El resultado sugiere que el tumor es maligno.')

    st.write(
        'Nota: Esta predicción es solo una estimación. Siempre consulte con un'
        ' profesional médico para un diagnóstico preciso.')


st.sidebar.header('Información general')
st.sidebar.write(
    'Esta aplicación utiliza un modelo de regresión logística para predecir'
    ' si un tumor de mama es benigno o maligno basándose en varias'
    ' características medidas.')
st.sidebar.write(
    'Por favor, ingrese los valores solicitados y haga clic en "Realizar'
    ' predicción" para obtener el resultado.')
