import streamlit
import requests
import json
        
def run():
    streamlit.title("Gemstone price prediction")
    carat = streamlit.text_input("carat")
    cut = str(streamlit.selectbox("cut", ['Ideal' , 'Premium', 'Very Good', 'Good', 'Fair']))
    color = str(streamlit.selectbox("color", ['G', 'E', 'F', 'H', 'D']))
    clarity = str(streamlit.selectbox("clarity", ['SI1', 'VS2', 'SI2', 'VS1', 'VVS2']))
    depth = streamlit.text_input("depth")
    table = streamlit.text_input("table")
    x = streamlit.text_input("x")
    y = streamlit.text_input("y")
    z = streamlit.text_input("z")

    data = {
                "carat": carat,
                "cut": cut,
                "color": color,
                "clarity": clarity,
                "depth": depth,
                "table": table,
                "x": x,
                "y": y,
                "z": z
            }
    
    if streamlit.button("Predict"):
        response = requests.post("http://localhost:8000/predict", json=data)
        prediction =response.text
        streamlit.success(f"The prediction from model: {prediction}")
    
if __name__ == '__main__':
    run()