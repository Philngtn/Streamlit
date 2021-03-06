import streamlit as st
from PIL import Image
import requests
import json



def app():
    st.write("""
    # Customer Sentiment Prediction using Amazon Web Services (SageMaker)
    This application predicts the **Customer Sentiment** based on product feedback!
    """)
    st.write('---')

    st.write('The application is using the latest **roBERTa** pre-trained model to predict the customer\'s sentiment.' )
    st.write('The API is generated by Amazon API Gateway and embedded to the page for the real-time predictions.')
    st.image("https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2018/05/17/load-test-sagemaker-3.gif", width=800)


    st.write('Feel free to enter the "customer feedback": ')
    fb = st.text_input('Enter the feebback')
    st.write('**Press enter** then click **predict**.')

    st.write('**Some examples:**')
    st.write('I love this product!, my dog always plays with it.')
    st.write('OK, but not great, it can be replaced my old tool.')
    st.write('This is not the right product. I want to refund!!.')



    if st.button('Predict'):
        api = {"features": [fb]}
        # print(type(api))
        # print(api)
        response = requests.post(st.secrets["AWS_API"],
                                 data=json.dumps(api))
        prediction = json.loads(json.loads(response.content)[0])
        label = prediction["predicted_label"]
        confidence = str(prediction["probability"])
        confidence = str(round(float(confidence)*100,2))


        if label == 1:
            st.write('The customer review seems to be **happy** with the confidence ' + confidence +'%.')
        elif label == 0:
            st.write('The customer review seems to be **neutral** with the confidence '+ confidence+'%.')
        elif label == -1:
            st.write('The customer review seems to be **unhappy** with the confidence '+ confidence+'%.')
        else:
            st.write('Server DOWNNNNN !!! Try again later ^^, or my wallet is empty :(, contact me if you want to see it run.')
    else:
        st.write('Please enter the review.')
