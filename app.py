from flask import Flask, request, render_template, redirect, send_file, send_from_directory
import os
import shutil

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    return render_template("home.html")

@app.route('/convertor', methods=['POST'])
def convertor():
    shutil.rmtree("Dataset")
    shutil.rmtree("Clinical")
    for i in request.files.getlist('Dataset'):
        path = i.filename.split('/')
        folder_path = "./Dataset"
        for j in range(1, len(path) - 1):
            folder_path = folder_path + "/" + path[j]

        if os.path.exists(folder_path) == False:
            os.makedirs(folder_path)
        folder_path = folder_path + "/" + path[-1]

        i.save(folder_path)

    for i in request.files.getlist('Clinical'):
        path = i.filename.split('/')
        folder_path = "./Clinical"
        for j in range(1, len(path) - 1):
            folder_path = folder_path + "/" + path[j]

        if os.path.exists(folder_path) == False:
            os.makedirs(folder_path)

        folder_path = folder_path + "/" + path[-1]

        i.save(folder_path)

    data_type = request.form.get('Type').lower()
    shutil.rmtree("BIDS")
    os.makedirs("BIDS")
    os.system("clinica convert " + data_type + "-to-bids ./Dataset ./Clinical ./BIDS")
    shutil.make_archive("bids", 'zip', "BIDS")
    
    return send_from_directory(directory="./", path="./", filename="bids.zip")

app.run();
