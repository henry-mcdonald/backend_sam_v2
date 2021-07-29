import os
import random
import json
import numpy as np
import pandas as pd
import pickle
import nltk
# Below line was needed once while running on Windows 10 and Ubuntu
nltk.download('punkt')
# Below line was needed once while running on Ubuntu
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model


# Paths to files
intents_file_path = 'data/intents.json'
all_data_pickle_file_path = 'models/all_data.pkl'
chatbot_model_file_path = 'models/chatbotmodel.h5'
epochs_log_file_path = 'models/epochs_log.csv'


def load_all_data_pickle():
    training = []
    try:
        with open(all_data_pickle_file_path, 'rb') as file:
            words, classes, training = pickle.load(file)
            # words, classes, training, labels = pickle.load(file)
    except OSError as oserr:
        print(f'[ERROR] : Pickle file could not be loaded: {all_data_pickle_file_path}')
        print(f'[ERROR] : {oserr}')
    return training

def create_all_data_pickle():
    lem = WordNetLemmatizer()
    intents = json.loads(open(intents_file_path).read())
    intents = intents['intents']
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']

    for intent in intents:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lem.lemmatize(word) for word in words if word not in ignore_letters]
    words = sorted(set(words))

    # pickle.dump(words, open('words.pkl', 'wb'))
    # pickle.dump(classes, open('classes.pkl', 'wb'))

    training = []
    output_empty = [0] * len(classes)

    # print(words)
    for doc in documents:
        bag = []
        word_patterns = doc[0]
        # print(doc[0])
        word_patterns = [lem.lemmatize(word.lower()) for word in word_patterns]
        # print(word_patterns)
        for word in words:
            if word in word_patterns:
                bag.append(1)
            else:
                bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    random.shuffle(training)
    training = np.array(training)
    with open(all_data_pickle_file_path, 'wb') as file:
        pickle.dump((words, classes, training), file)
        # pickle.dump((words, classes, training[:, 0], training[:, 1]), file)
    
    print('[INFO] :   - Pickle created from JSON')
    return training


def create_keras_model(training):
    """
        Trains the NN from the intents.json file
    """
    # train / test
    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    ## model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    # compile
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # train
    hist = model.fit(train_x, train_y, epochs=300, batch_size=5, verbose=1)
    # pd.DataFrame(hist.history).to_csv(epochs_log_file_path)
    model.save(chatbot_model_file_path, hist)
    print('[INFO] :   - Model trained from Pickle')


def create_training_data():
    """
        Delete all files in "Models" to retrain model from updated intents.json
        Note this uses f-strings and requires Python v3.6 or higher
    """
    training_array = []
    pickle_ceated = True
    pline = '\n-----------------------------------\n'
    print(pline)
    
    # CHECK FOR PICKLE FILE
    if(os.path.exists(all_data_pickle_file_path)):
        print(f'[INFO] : Pickle file exists: {all_data_pickle_file_path}')
        print(f'[INFO] : Loading Pickle file...')
        try:
            training_array = load_all_data_pickle()
            print(f'[INFO] : Pickle file loaded.')
        except:
            print(f'[ERROR] : PICKLE FILE COULD NOT BE LOADED')
    else:
        print(f'[INFO] : Pickle file DOES NOT EXIST')
        
        # CHECK FOR JSON FILE
        print(f'[INFO] : Need JSON file to create Pickle File')
        if(os.path.exists(intents_file_path)):
            print(f'[INFO] : JSON file exists: {intents_file_path}')
            print(f'[INFO] : Creating Pickle file...')
            try:
                training_array = create_all_data_pickle()
                print(f'[INFO] : Pickle File created: {all_data_pickle_file_path}')
            except OSError as oserr:
                print(f'[ERROR] : PICKLE FILE COULD NOT BE CREATED')
                print(f'[ERROR] : {oserr}')
        else:
            print(f'[ERROR] : JSON FILE AND PICKLE FILE DO NOT EXIST.')
            print(f'[ERROR] : CANNOT CREATE KERAS MODEL WITHOUT AT LEAST ONE OF THESE FILES.\n')
            print(f'[INFO] : EXITING.')
            pickle_ceated = False
    print(pline)

    if(pickle_ceated):
        # CHECK FOR KERAS MODEL FILE
        if(os.path.exists(chatbot_model_file_path)):
            print(f'[INFO] : Keras Model file exists: {chatbot_model_file_path}')
            print(f'')
        else:
            print(f'[INFO] : Keras Model file DOES NOT EXIST')
            print(f'[INFO] : Creating Keras Model file...')
            try:
                create_keras_model(training_array)
                print(f'[INFO] : Keras Model file created: {chatbot_model_file_path}')
            except IndexError as ind:
                print(f'[ERROR] : KERAS MODEL FILE COULD NOT BE CREATED')
                print(f'[ERROR] : INDEX ERROR: {ind}')
        print(pline)

if __name__ == "__main__":
    """
        Delete all files in "Models" to retrain model from updated intents.json
        Note this uses f-strings and requires Python v3.6 or higher
    """
    create_training_data()
