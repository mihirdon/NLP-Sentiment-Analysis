#from nltk.sem.relextract import list2sym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import re
import nltk
import sklearn
import keras
import tensorflow as tf
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras import utils

#VISUALIZING DATA ____________________________________________________________________________________
data = pd.read_csv('/Users/mihir/Dropbox/My Mac (macbook-pro.lan)/Documents/training.1600000.processed.noemoticon.csv', encoding= 'latin', header=None)
#visualization of data
#print(data.head(10))
#print(data.tail(10))

data.columns = ['polarity', 'ids', 'date', 'flag', 'user', 'text']

#change polarity to be 0 and 1 (0 is negative, 1 is positive)
data['polarity'] = data['polarity'] / 4

#remove date, flag, and id columns not very useful
data.drop(columns= ['date', 'flag', 'ids'], inplace=True)
#print(data['polarity'])
#print(data.tail(10))

#DONE VISUALIZING DATA ______________________________________________________________________________________--


#ORGANIZING TEXT ___________________________________________________________________________________________
stop = " | ".join(stopwords.words('english'))
links = '@\S+|https?:\S+|http?:\S+|[^a-z0-9]+'

stemmer = SnowballStemmer('english')
def preprocess(sentence):
    #remove all links and special chars(sub links with " ") AND lowercase(.lower()) AND remove all leading and trailing spaces(.strip()) 
    sentence = re.sub(links, " ", sentence.lower()).strip()
    #output = ""
    #list_output = []

    result = [stemmer.stem(word) for word in sentence.split() if word not in stop]
    return " ".join(result)

#ONLY DO WHEN DONE TESTING
#data['text'] = data['text'].apply(preprocess)
#testData = data['text'][0: 10]
#testDataPolarity = data['polarity'][0:10]
#testData = testData.apply(preprocess)
data['text'] = data['text'].apply(preprocess)


#ORGANIZING TEXT DONE____________________________________________________________________________________________




#MAKING DATA SEQUENCE REP ________________________________________________________________________________________________________________________

#Splitting up data into training set (70%) and test set(30%)     testData     testDataPolarity
training_data, test_data, training_y, test_y = train_test_split(data['text'], data['polarity'], test_size= 0.3, random_state= 7)

#print(training_data)
#print(training_y)

#Setting up tokenizer (tokenizer will convert words into numbers based off of internal library thus allowing us to send it in as a vector)
token = Tokenizer()
token.fit_on_texts(training_data)  #updating the tokenizer's library to fit the words we have
token.fit_on_texts(test_data)
vocab_size = len(token.word_index) + 1 #number of words tokenizer knows (current word_index is always set to # of words - 1 hence the + 1)
max_length_of_tweets = 40 #the maximum number of words our tokenizer will allow (will act as the number of parameters as well)


tokenized_training = token.texts_to_sequences(training_data) #tokenizing the data
tokenized_test = token.texts_to_sequences(test_data)
X_training_data = pad_sequences(tokenized_training, maxlen=max_length_of_tweets, padding= 'post') #padding the ends of the sequences generated to fit max # of words
X_test_data = pad_sequences(tokenized_test, max_length_of_tweets, padding='post')
print("done data seq")
#MAKING DATA SEQUENCE DONE ___________________________________________________________________________________________________________________



#GIVING WORDS MEANING____________________________________________________________________________________________________________________

# The glove dictionary is a vectorized representation of 6 billion words accrued by Stanford, each word used machine learning to 
# create a "neural network" of parameters to represent its meaning via numbers, that is what we are downloading
# Goal: grant every word in the vocab_library their respective nn

glove_dictionary = dict()
embedded_layer_size = 100 #why not, first try anyway

glove_file = open('/Users/mihir/Dropbox/My Mac (macbook-pro.lan)/Downloads/glove.6B.100d.txt')

# Retrieving the data from the glove file 
for line in glove_file:
    line_list = line.split()
    glove_dictionary[line_list[0]] = np.asarray(line_list[1:])

glove_file.close() #we got all the data we need

# Applying the data to the words
embedded_matrix = np.zeros((vocab_size, embedded_layer_size))
for word, index in token.word_index.items():
    if word in glove_dictionary: #checking if word exists in the dictionary
        embedded_matrix[index] = glove_dictionary.get(word) #since embedded_matrix has all vocab_size it will refer to the same spot as word_index
                                                            #thus keeping both representations of the dictionary

print("done glove")
#GIVING WORDS MEANING DONE________________________________________________________________________



#MAKING NEURAL NETWORK_____________________________________________________________________________________________

#embedded layer of input: # of tweets x max_length_of_tweets output: # of tweets x max_length x 100 (embedded layer size)
embedding_layer = tf.keras.layers.Embedding(vocab_size, embedded_layer_size, 
input_length= max_length_of_tweets, weights= [embedded_matrix], trainable= False)

num_epochs = 10 #orignially 5 but had underfitting, then 7, then 9
batch_size = 2000

model = Sequential()
model.add(embedding_layer)
model.add(tf.keras.layers.Dense(256, activation= 'sigmoid')) #og 128, then 256: 256 output parameters why not, activation 'relu' also why not
model.add(tf.keras.layers.Dropout(0.05)) #reduce overfitting
model.add(tf.keras.layers.Dense(64, activation= 'sigmoid')) #og 32, then 64, then 256
model.add(tf.keras.layers.Dropout(0.05))
model.add(Flatten())
model.add(tf.keras.layers.Dense(1, activation= 'sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

model_data = model.fit(X_training_data, training_y, batch_size, num_epochs, validation_data= (X_test_data, test_y))

#GRAPHS for fun and so we can go back and change stuff
plt.figure(figsize=(10,5))
plt.plot(model_data.history['accuracy'])
plt.plot(model_data.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train accuracy', 'Test accuracy'], loc='lower right')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(model_data.history['loss'])
plt.plot(model_data.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train loss', 'Test loss'], loc='upper right')
plt.suptitle('Accuracy and loss for second model')
plt.show()
#print(testData)