import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def app():
    st.write("""
    # California House Price Prediction
    This application predicts the **California House Price**!
    """)
    st.write('---')

    # Loads the Cali House Price Dataset
    cali = datasets.fetch_california_housing()
    X = pd.DataFrame(cali.data, columns=cali.feature_names)
    Y = pd.DataFrame(cali.target, columns=["MedHouseVal"])

    # Sidebar
    # Header of Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    def user_input_features():
        # Median income in block group
        MEDINC = st.sidebar.slider('Median Income (expressed in hundreds of thousands of dollars)',float(X.MedInc.min()) ,float(X.MedInc.max()),float( X.MedInc.mean()))
        AGE = st.sidebar.slider('House Age',int(X.HouseAge.min()) , int(X.HouseAge.max()), int(X.HouseAge.mean()) )
        AVEROOMS = st.sidebar.slider('Average Rooms',int(X.AveRooms.min()) ,int(X.AveRooms.max()) , int(X.AveRooms.mean()) )
        AVEBEDRMS = st.sidebar.slider('Average Bedrooms',int( X.AveBedrms.min()), int(X.AveBedrms.max()) , int( X.AveBedrms.mean()))
        POP = st.sidebar.slider('Population', int(X.Population.min()) ,int(X.Population.max()) , int( X.Population.mean()))
        AVEOCCUP = st.sidebar.slider('Average Occupances', int(X.AveOccup.min()) , int(X.AveOccup.max()) , int(X.AveOccup.mean()) )
        LATITUDE = st.sidebar.slider('Latitude',float(X.Latitude.min()) ,float( X.Latitude.max()),float( X.Latitude.mean()))
        LONGTITUDE = st.sidebar.slider('Longitude',float(X.Longitude.min()) ,float( X.Longitude.max()),float(X.Longitude.mean()) )

        data = {'MedInc': MEDINC,
                'HouseAge': AGE,
                'AveRooms': AVEROOMS,
                'AveBedrms': AVEBEDRMS,
                'Population': POP,
                'AveOccup': AVEOCCUP,
                'Latitude': LATITUDE,
                'Longitude': LONGTITUDE
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()

    # Main Panel

    # Print specified input parameters
    st.write('Specified Input parameters')
    st.write(df)
    st.write('---')

    # Build Regression Model
    model = RandomForestRegressor()
    model.fit(X, Y.values.ravel())
    # Apply Model to Make Prediction
    prediction = model.predict(df)


    st.write('Prediction of the median house value for California districts')
    st.write('The price would around **{}** dollars'.format(round(float(prediction)*100000,3)))
    st.write('---')

    st.write('Preview of the location')


    map = pd.DataFrame(
         np.random.randn(1, 2) / [1,1] + [float(df.Latitude), float(df.Longitude)],
         columns=['lat', 'lon'])

    st.map(map)

# Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)
#
# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')
#
# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
