import streamlit as st
import pickle as pk
import numpy as np
import requests
import streamlit_lottie as st_lottie


Columns = pk.load(open("columns.pkl","rb"))
Columns_name = pk.load(open("columns_name.pkl","rb"))



st.subheader("Welcome to Mushroom Classification :wave:")
name = st.title("Mushroom_Classification")
st.write("Mushrooms have different characteristics such as color, shape, odor, and gill size, among others, which can be used to classify them into edible or poisonous categories. ")

## Animation


# def animation(url):
#       r = requests.get(url)
#       if r.status_code != 200:
#             return None
#       return r.json()
# Lottie_coding = animation("https://assets2.lottiefiles.com/packages/lf20_3rwasyjy.json")
# st_lottie(Lottie_coding , height=300 ,key = "as")



st.write("---")
predict = []
# print(Columns)
import streamlit as st
for i in range(1,len(Columns)):
        option = st.selectbox(
            Columns_name[i],
            Columns[i])
        predict.append(np.where( Columns[i]==option))
List  =   np.concatenate(predict,axis = 0) 
# st.write('You selected:',List)
l = [item for sublist in List for item in sublist]

Input = np.array(l)

log_reg = pk.load(open("columns_name.pkl","rb"))
pk.dump(Input,open("predi.pkl","wb"))

def recommend(L):
      pred = log_reg.predict(Input.reshape(1, -1))
      return pred


st.write("---")
st.write("###")
if st.button('Predicted Result'):
    pk.dump(Input,open("predi.pkl","wb"))
    Out=pk.load(open("Final.pkl","rb"))
    st.write(Out[0])