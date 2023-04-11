from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tokenize
import pandas as pd
from keras.models import load_model
import numpy as np
def home(request):
    return render(request,'home.html')


model = load_model('app\poem_generator.h5')



with open('app/robert_frost.txt',encoding="utf8") as story:
  story_data = story.read()

# print(story_data)

# data cleaning process
import re                                # Regular expressions to use sub function for replacing the useless text from the data

def clean_text(text):
  text = re.sub(r',', '', text)
  text = re.sub(r'\'', '',  text)
  text = re.sub(r'\"', '', text)
  text = re.sub(r'\(', '', text)
  text = re.sub(r'\)', '', text)
  text = re.sub(r'\n', '', text)
  text = re.sub(r'“', '', text)
  text = re.sub(r'”', '', text)
  text = re.sub(r'’', '', text)
  text = re.sub(r'\.', '', text)
  text = re.sub(r';', '', text)
  text = re.sub(r':', '', text)
  text = re.sub(r'\-', '', text)

  return text

# cleaning the data
# lower_data = story_data.lower()           # Converting the string to lower case to get uniformity

split_data = story_data.splitlines()      # Splitting the data to get every line seperately but this will give the list of uncleaned data

# print(split_data)                         

final = ''                                # initiating a argument with blank string to hold the values of final cleaned data

for line in split_data:
  line = clean_text(line)
  final += '\n' + line

# print(final)

final_data = final.split('\n')       # splitting again to get list of cleaned and splitted data ready to be processed
# print(final_data)



# # Instantiating the Tokenizer
max_vocab = 1000000
tokenizer = Tokenizer(num_words=(max_vocab))
tokenizer.fit_on_texts(final_data)

def predict_words(seed, no_words):
  for i in range(no_words):
    
    token_list = tokenizer.texts_to_sequences([seed])[0]
    token_list = pad_sequences([token_list], maxlen=11-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=1)

    new_word = ''

    for word, index in tokenizer.word_index.items():
      if predicted == index:
        new_word = word
        break
    seed += " " + new_word
    
  return seed

def pred(request):
    if request.method == 'POST':
        poem = request.POST.get('poem')
        # do something with the poem data
        next_words = 30

        res=predict_words(poem, next_words)
        print(res)
        context={'predicted_text': res}
        return render(request, 'home.html', context)

    return render(request, 'home.html')
   