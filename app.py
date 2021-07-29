# This script runs the chat bot in a tkinter GUI

import datetime
import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import load_model

#INITIALIZE - for first run only:
# import subprocess
# subprocess.call(['python', 'train.py'])
# OR 
import train
train.create_training_data()

import nltk
# Below line was needed once while running on Windows 10 and Ubuntu
# nltk.download('punkt')
# Below line was needed once while running on Ubuntu
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

# import time
from flask import Flask, render_template  # , Response
from flask_socketio import SocketIO
# from flask.ext.socketio import SocketIO, emit

# Files and Paths
intents_file_path = 'intents.json'
all_data_pickle_file_path = './models/all_data.pkl'
chatbot_model_file_path = './models/chatbotmodel.h5'


def log_dt(log_level = 'INFO'):
    return datetime.datetime.now().isoformat() + ' [' + log_level.upper() + '] : '


class ChatBot:
    
    def __init__(self, bot_name, intents_file_path, all_data_pickle_file_path, chatbot_model_file_path):
        self.words, self.classes, self.training = self.load_pickle_file(all_data_pickle_file_path)
        self.model = load_model(chatbot_model_file_path)
        self.intents = json.loads(open(intents_file_path).read())
        self.bot_name = bot_name
    
    def load_pickle_file(self, pkl_file_path):
        with open(pkl_file_path, 'rb') as file:
            words, classes, training = pickle.load(file)
        return words, classes, training

    def clean_sentence(self, sentence):
        lem = WordNetLemmatizer()
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lem.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        predictions = self.model.predict(np.array([bow]))[0]
        error_thresh = 0.25
        results = [[i, r] for i, r in enumerate(predictions) if r > error_thresh]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, msg):
        intents_list = self.predict_class(msg)
        tag = intents_list[0]['intent']
        prob = float(intents_list[0]['probability'])
        list_of_intents = self.intents['intents']
        error_thresh = 0.60
        if prob > error_thresh:
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    break
        # if the error_thresh is not met respond IDK
        else:
            # the first item in intents should be responses to unknown inputs
            result = random.choice(self.intents['intents'][0]['responses'])
        return result


class FlaskWindow:

    def __init__(self):
        self.convo = []

    def reply_message(self, msg, sender):
        if(not msg):
            return
        self.convo.append(f'{sender}: {msg}\n\n')


app = Flask(__name__)
# change this and make it pull from a file so it does not pull to github
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route('/', methods=['POST', 'GET'])
def sessions():
    return render_template('index.html') #, form=form, chat_responses=chat_responses)


def messageReceived(methods=['GET', 'POST']):
    print('message was recieved.')


def messageSent(methods=['GET', 'POST']):
    print('message was sent.')


@socketio.on('my event')
def handle_my_custom_event(json_msg, methods=['GET', 'POST']):
    print(f'{log_dt()} RCVD: {str(json_msg)}')
    socketio.emit('my response', json_msg, callback=messageReceived)
    
    print(f'\n\nTRY EXTRACT MESSAGE FROM: {json_msg}')
    get_msg = json_msg['message']
    send_msg = cb.get_response(get_msg)
    print(f'TRY EXTRACT MESSAGE : {send_msg}\n')
    mock_data = {'user_name': cb.bot_name.upper(), 'message': send_msg}

    sam_json = json.dumps(mock_data, sort_keys=True)
    socketio.emit('sam response', sam_json, callback=messageSent)
    print(f'{log_dt()} SENT: {str(sam_json)}')

# from flask_sslify import SSLify
# if 'DYNO' in os.environ: # only trigger SSLify if the app is running on Heroku
#     sslify = SSLify(app)
if(__name__ == '__main__'):
    bot_name = 'Sam'
    cb = ChatBot(bot_name, intents_file_path,
                 all_data_pickle_file_path, chatbot_model_file_path)
    from waitress import serve
    serve(app, host='0.0.0.0', port=8000)
