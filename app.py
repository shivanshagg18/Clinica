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

@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/convertor5', methods=['POST'])
def convertor5():
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

    os.system("./fsl_affine_img_reg.sh")
    shutil.make_archive("caps", 'zip', "CAPS")
    return send_from_directory(directory="./", path="./", filename="caps.zip")

@app.route('/train')
def train():
    return render_template("train.html")

@app.route('/convertor6', methods=['POST'])
def convertor6():
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


    model = request.form.get('model')
    optimizer = request.form.get('optimizer')
    learning_rate = request.form.get('learning_rate')
    train_bs = request.form.get('train_bs')
    val_bs = request.form.get('val_bs')
    train_epochs = request.form.get('train_epochs')
    split_ratio = request.form.get('split_ratio')
    num_workers = request.form.get('num_workers')


    shutil.rmtree("Model")
    os.makedirs("Model")
    os.system("python train_torchio.py --clinica_CAPS_path ./CAPS --output_model_path ./Model --model " + model + " --optimizer " + optimizer + " --learning_rate " + learning_rate + " --train_bs " + train_bs + " --val_bs " + val_bs + " --train_epochs " + train_epochs + " --split_ratio " + split_ratio + " --num_workers " + num_workers)
    shutil.make_archive("model", 'zip', "Model")
    return send_from_directory(directory="./", path="./", filename="model.zip")

app.run();