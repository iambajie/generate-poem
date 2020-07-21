# web显示，并保存生成结果

import os
from poem_writer import PoemWrite, start_model
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
import time
import datetime
import csv

app = Flask(__name__)

writer = start_model()


@app.route('/', methods=['POST', 'GET'])
def write_poem():
    start_with = ''
    poem_style = 0

    startwith = request.form.get('start_with')
    poemstyle = request.form.get('poem_style')
    numsentence = request.form.get('num_sentence')
    text = []

    start_with = startwith
    poem_style = poemstyle
    num_sentence = numsentence
    print(poem_style)
    print(numsentence)
    print(start_with)

    if start_with:
        if poem_style == '2':
            text = writer.cangtou(start_with)

    if poem_style == '1':
        text = writer.free_verse(numsentence)

    print(text)

    nowTime = datetime.datetime.now().strftime('%Y-%m-%d')

    with open(r"./save/%s.csv" % nowTime, 'a', newline='') as f:
        ff = csv.writer(f)
        ff.writerow([poem_style, num_sentence, start_with, text])

    return render_template('index.html', choosetext=text)


if __name__ == "__main__":
    app.run()