import streamlit as st
import pickle 
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

ps=PorterStemmer()



def textpreprocess(txt):
    
    text=txt.lower()
    text=nltk.word_tokenize(text)
    newlist=[]
    
    for word in text:
        if word.isalnum():
            
            newlist.append(word)
            
    newlist2=[]        
            
    
    
    for word in newlist:
        
        if word not in stopwords.words('english') and word not in string.punctuation:
            newlist2.append(word)
            
     
    
    newlist3=[]
    for word in newlist2:
        
        newlist3.append(ps.stem(word))
        
    return " ".join(newlist3)  

### preprocess ---> vectorization-->model apply -->output
tfidf=pickle.load(open("vectorizer.pkl","rb"))
model=pickle.load(open("model.pkl","rb"))


st.title("SMS Spam Classifier")
input_sms=st.text_area("Enter the message here")

if st.button("Predict"):

    

    transformed_sms=textpreprocess(input_sms)


    vector_input=tfidf.transform([transformed_sms])

    result=model.predict(vector_input)[0]


    if result==1:
       st.header("Spam")
    else:
       st.header("Not Spam")    
