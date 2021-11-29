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

@app.route('/freesurfer')
def freesurfer():
    return render_template("freesurfer.html")

@app.route('/convertor2', methods=['POST'])
def convertor2():
    shutil.rmtree("BIDS")
    for i in request.files.getlist('BIDS'):
        path = i.filename.split('/')
        folder_path = "./BIDS"
        for j in range(1, len(path) - 1):
            folder_path = folder_path + "/" + path[j]

        if os.path.exists(folder_path) == False:
            os.makedirs(folder_path)
        folder_path = folder_path + "/" + path[-1]

        i.save(folder_path)

    shutil.rmtree("CAPS")
    os.makedirs("CAPS")
    os.system("clinica run t1-freesurfer ./BIDS ./CAPS")
    shutil.make_archive("caps", 'zip', "CAPS")
    
    return send_from_directory(directory="./", path="./", filename="caps.zip")

@app.route('/rotate')
def rotate():
    return render_template("rotate.html")


@app.route('/convertor3', methods=['POST'])
def convertor3():
    shutil.rmtree("CAPS")
    for i in request.files.getlist('CAPS'):
        path = i.filename.split('/')
        folder_path = "./CAPS"
        for j in range(1, len(path) - 1):
            folder_path = folder_path + "/" + path[j]

        if os.path.exists(folder_path) == False:
            os.makedirs(folder_path)
        folder_path = folder_path + "/" + path[-1]

        i.save(folder_path)

    os.system("python fs_brain_mgz_to_npy.py --clinica_CAPS_path ./CAPS")
    shutil.make_archive("caps", 'zip', "CAPS")
    return send_from_directory(directory="./", path="./", filename="caps.zip")

@app.route('/nifti')
def nifti():
    return render_template("nifti.html")

@app.route('/convertor4', methods=['POST'])
def convertor4():
    shutil.rmtree("CAPS")
    for i in request.files.getlist('CAPS'):
        path = i.filename.split('/')
        folder_path = "./CAPS"
        for j in range(1, len(path) - 1):
            folder_path = folder_path + "/" + path[j]

        if os.path.exists(folder_path) == False:
            os.makedirs(folder_path)
        folder_path = folder_path + "/" + path[-1]

        i.save(folder_path)

    os.system("python npy_to_nii.py --clinica_CAPS_path ./CAPS")
    shutil.make_archive("caps", 'zip', "CAPS")
    return send_from_directory(directory="./", path="./", filename="caps.zip")

app.run();