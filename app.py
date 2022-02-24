import streamlit as st
import os
import sys
sys.path.append( '.' )

# Custom imports
from multipages import MultiPage
from CaliHousePrice import caliHousePrice
from roBERTa_AWS_APIs import roBERTa_AWS

# Create an instance of the app
app = MultiPage()

# Title of the main page
st.write("Data Science and ML Engineer Application")

# Add all your applications (pages) here
app.add_page("NLP with Amazon SageMaker", roBERTa_AWS.app)
app.add_page("California House Price", caliHousePrice.app)

# The main app
app.run()
