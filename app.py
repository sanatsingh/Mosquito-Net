from flask import Flask, request, render_template, send_from_directory, send_file
from inference import get_result
import os

app = Flask(__name__)
classes=['Infected','Uninfected']

@app.route('/')
def index():
        return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return render_template('analyze.html')

    if request.method == 'POST':
        if 'file' not in request.files:
            print('file not uploaded')
            return
        im_file = request.files['file']
        img = im_file.read()
        result = get_result(image_bytes=img)
        if(result in classes):
            return render_template('result.html', result=result)
        else :
            return render_template('index.html')

@app.route('/samples_download')
def samples_download():
    return send_from_directory(directory='samples',filename='samples.zip',as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0')


