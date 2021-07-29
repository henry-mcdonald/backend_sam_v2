# This script runs the chat bot in a tkinter GUI

import json
import random
import pickle
import numpy as np
from tensorflow.keras.models import load_model

import nltk
# nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

import time
import tkinter as tk

# GLOBAL VARS
# Files and Paths
intents_file_path = './data/intents.json'
all_data_pickle_file_path = './models/all_data.pkl'
chatbot_model_file_path = './models/chatbotmodel.h5'



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


class TkWindow:
    
    BG_GRAY = "#ABB2B9"
    BG_COLOR = "#17202A"
    TEXT_COLOR = "#EAECEE"
    FONT = "Helvetica 14"
    FONT_BOLD = "Helvetica 13 bold"

    def __init__(self, chat_bot):
        self.window = tk.Tk()
        self._setup_main_window()
        self.chat_bot = chat_bot
    
    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=self.BG_COLOR)

        # head label
        head_label = tk.Label(self.window, bg=self.BG_COLOR, fg=self.TEXT_COLOR, text="Welcome", font=self.FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = tk.Label(self.window, width=450, bg=self.BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = tk.Text(self.window, width=20, height=2, bg=self.BG_COLOR, fg=self.TEXT_COLOR, font=self.FONT, padx=5, pady=5)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=tk.DISABLED)

        # scroll bar
        scrollbar = tk.Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.984)
        scrollbar.configure(command=self.text_widget.yview)

        # bottom label
        bottom_label = tk.Label(self.window, bg=self.BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # message entry box
        self.msg_entry = tk.Entry(
            bottom_label, bg="#2C3E50", fg=self.TEXT_COLOR, font=self.FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06,
                             rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # send button
        send_button = tk.Button(bottom_label, text="Send", font=self.FONT_BOLD, width=20, bg=self.BG_GRAY, command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):  # event will take the argument entry
        msg = self.msg_entry.get()       # get input text as string
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        self.msg_entry.delete(0, tk.END)  # delete msg from txt_entry window
        msg1 = f"{sender}: {msg}\n\n"     # The users message
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg1)
        self.text_widget.configure(state=tk.DISABLED)

        # ints = self.chat_bot.predict_class(msg)
        res = self.chat_bot.get_response(msg)
        # time.sleep(1)
        time.sleep(0.1)
        msg2 = f"{self.chat_bot.bot_name}: {res}\n\n"
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, msg2)
        self.text_widget.configure(state=tk.DISABLED)

        self.text_widget.see(tk.END)


if __name__ == "__main__":
    bot_name = 'Sam'
    cb = ChatBot(bot_name, intents_file_path,
                  all_data_pickle_file_path, chatbot_model_file_path)
    win = TkWindow(cb)
    win.run()

