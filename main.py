import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import ToTensor
from flask import Flask,render_template,request ,jsonify
from werkzeug.utils import secure_filename
import os
import torch.nn as nn

target_dict={0.0: 'Parasitized',
             1.0: 'Uninfected'}

model = torch.load("models/models.pt",map_location ='cpu')
transform=transforms.Compose([ToTensor(),transforms.Resize((224,224)),])
lname_file = ['jpeg','jpg','png']

app = Flask(__name__, template_folder='templates')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_FOLDER = os.path.join('static', 'IMG')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def upload():
    pred = []
    return render_template('index.html',pred=pred,len=len(pred))

def transform_img(path):
    image = Image.open(path).convert('RGB')
    image_transform = transform(image).cpu()
    image_transform.size()
    image_transform = image_transform.unsqueeze(0)
    return image_transform

def predict(img):
    output = model(img)
    _, preds = torch.max(output, 1)
    prob = nn.Softmax(dim=1)(output)
    percent = torch.max(prob, 1).values
    return [str(target_dict[float(preds.cpu().numpy())])],percent.detach().numpy()[0]

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    result_json = {}
    pred = []
    if request.method == 'POST':
        f = request.files['file']
        org_filename = f.filename
        if org_filename.split(".")[-1].lower() not in lname_file:
                return render_template('index.html',pred="",check_file = False)
        elif org_filename.split(".")[-1].lower() in lname_file:
            
            path = os.path.join(app.config['UPLOAD_FOLDER'],org_filename)
            f.save(path)
            
            img = transform_img(path)
            pred,percent = predict(img)
            return render_template('index.html',pred=pred,percent=round(percent*100,2),filename=f.filename,filepath=path,check_file = True,len=len(pred))
    else:
        return render_template('index.html',pred=pred,len=len(pred)) 


if __name__=="__main__":
    app.run(debug=True,port=5050)