from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
import gensim
from gensim.models.fasttext import FastText
import pandas as pd


col=['col1','col2','col3','col4','col5']
data= pd.read_csv("rdany.csv",encoding = "ISO-8859-1", header=None, names=col)
data.drop(['col1','col3','col4','col5'],axis=1,inplace=True)
k=[]
for sent in data['col2']:
    k.append(sent)
import re
dat=[]
for sen in data['col2']:
    wordList = str(sen).split()
    dat.append(wordList)

model = FastText.load("fasttext2.model")
model.build_vocab(dat, update=True)
model.train(dat, total_examples=len(dat), epochs=100)

data_words=['hello','hi','what']

interpreter = RasaNLUInterpreter('models/current/nlu')
agent = Agent.load('models/dialogue', interpreter=interpreter)

print("Your bot is ready to talk! Type your messages here or send 'stop'")

def proc(text):
    st=""
    for word in range(len(text)):
        corr=model.wv.most_similar(text[word], topn=5)
        temp=0
        for i in range(len(corr)):
            for j in range(len(data_words)):
                if corr[i][0]==data_words[j]:
                    text[word]=corr[i][0]
                    temp=1
                    break
            if temp==1:
                break
        if word!=len(text)-1:
            st=st+text[word]+" "
        else:
            st=st+text[word]
    print("ok\n")
    return st


while True:
    a = input()
    k=a.split()
    a=proc(k)
    if a == 'stop':
        break
    responses = agent.handle_message(a)
    for response in responses:
        print(response["text"])
        
