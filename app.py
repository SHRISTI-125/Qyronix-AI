from flask import (
    Flask,
    render_template,
    request,
    send_file,
    redirect,
    url_for,
    session
)
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus.flowables import HRFlowable

from datetime import datetime

import os

from PIL import Image

import torch
import torch.nn.functional as F

from torchvision import transforms
from torchvision import models

from werkzeug.utils import secure_filename

from geopy.geocoders import Nominatim

import cv2
import numpy as np
import os
import json
import math

from datetime import datetime




#app of flask

app = Flask(__name__)

app.secret_key = "qyronix_secret"

#device

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

#geolocator; for finding nearest hospital from db

geolocator = Nominatim(
    user_agent="qyronix_ai"
)

#result is global

last_result = {}

#models of diff. disease

MODELS = {

    "retinopathy": {

        "path": "models/dr_model.pth",

        "classes": [
            "mild",
            "moderate"
        ]
    },

    "brain_tumor": {

        "path": "models/brain_model.pth",

        "classes": [
            "glioma",
            "meningioma",
            "pituitary"
        ]
    },

    "skin_cancer": {

        "path": "models/skin_model.pth",

        "classes": [
            "benign",
            "malignant"
        ]
    }
}

#model cache

loaded_models = {}

#transform images, taki sb ek jaise img ho; easy for training
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])



#app.secret_key = "qyronix_secret_key"


#logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

#loading model..
def load_model(disease):
    if disease in loaded_models: #if disease here only 3, anyone is there in model, return it
        return loaded_models[disease]

    config = MODELS[disease]

    num_classes = len(
        config["classes"] 
    ) # like skin_cancer have bengin and malignant

    model = models.resnet18(
        weights="DEFAULT"
    ) #resnet is lightweight model and loads fast

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(
            model.fc.in_features,
            num_classes
        )
    )# in final classification=>sequential model has dropout to remove 50%, so that it will not overfit

    model.load_state_dict(
        torch.load(
            config["path"],
            map_location=device
        )
    )# load is saved as .pth

    model.to(device)

    model.eval()#Dropout automatically becomes inactive

    loaded_models[disease] = model

    return model


#prediction
def predict(img_path, disease):
    config = MODELS[disease]
    model = load_model(disease)
    classes = config["classes"]

    img = Image.open(img_path).convert("RGB") #opening images

    x = transform(img).unsqueeze(0).to(device)# normalize=used to standardize pixel values, unsqueeze for adding batch dimensions

    with torch.no_grad():
        out = model(x) #forward pass
        prob = F.softmax(out, dim=1) #logits into probability
        conf, pred = torch.max(prob, 1) #confidence and predicted class

    confidence = conf.item() * 100

    # confidence
    confidence = min(confidence, 95.6)
    return (
        classes[pred.item()],
        confidence,
        model
    )


#gradcam; to focus on important area
def generate_gradcam(img_path, model):
    gradients = []
    activations = []

    def fw_hook(m, i, o): #forward pass, layer, input, output
        activations.append(o) #activations at different layer

    def bw_hook(m, gi, go): #backpropagation
        gradients.append(go[0])

    layer = model.layer4[-1] #last layer of ResNet
    h1 = layer.register_forward_hook(fw_hook) #register fwd pass and fwd hook
    h2 = layer.register_full_backward_hook(bw_hook)

    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    out = model(x)
    cls = out.argmax(dim=1) #old gradients
    model.zero_grad()
    out[0, cls].backward() #preventing old gradients from mixing
    grads = gradients[0].detach().cpu().numpy()[0]
    acts = activations[0].detach().cpu().numpy()[0]

    weights = np.mean(
        grads,
        axis=(1,2)
    )#extracting features

    cam = np.zeros(
        acts.shape[1:],
        dtype=np.float32
    )#compute channel weight
    #Each feature map receives importance weight

    #formula of important feature map
    for i, w in enumerate(weights):
        cam += w * acts[i]
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224,224))
    cam = (cam - cam.min()) / (cam.max() + 1e-8) #prevent error

    binary = cam > 0.6
    coords = np.column_stack(
        np.where(binary)
    )

    original = cv2.imread(img_path)
    original = cv2.resize(
        original,
        (224,224)
    )

    if len(coords) > 0:
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        cv2.rectangle(
            original,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )
    result = heatmap * 0.4 + original #heatmap
    base, ext = os.path.splitext(img_path)
    cam_path = base + "_cam" + ext
    cv2.imwrite(cam_path, result)
    h1.remove()
    h2.remove()
    return cam_path


#location; coordinates
def get_coordinates(location_name):
    try:
        location = geolocator.geocode(location_name)
        if location:
            return (location.latitude, location.longitude)
    except:
        pass
    return None, None

#distance
def calculate_distance(lat1, lon1, lat2, lon2):
    return math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)


#nearest hospital
def get_nearest_hospitals(disease,user_lat,user_lon):
    with open("database/hospitals.json") as f:
        hospital_data = json.load(f)
    matched = []

    for item in hospital_data:
        if item["disease"] == disease:
            for hosp in item["recommended_hospitals"]:
                dist = calculate_distance(
                    user_lat,
                    user_lon,
                    hosp["latitude"],
                    hosp["longitude"]
                )
                hosp["distance"] = dist
                matched.append(hosp) # on basis of location
    matched = sorted(
        matched,
        key=lambda x: x["distance"]
    )
    return matched[:3] #top 3 returned


#pdf
def generate_pdf():
    global last_result
    if "name" not in last_result:
        return None
    file_path = "report.pdf"
    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=20
    )#defining pdf look

    styles = getSampleStyleSheet()
    content = []

    #custom style
    title_style = ParagraphStyle(
        "title",
        parent=styles["Heading1"],
        fontSize=26,
        leading=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=20
    )
    heading_style = ParagraphStyle(
        "heading",
        parent=styles["Heading2"],
        fontSize=16,
        textColor=colors.HexColor("#2563eb"),
        spaceAfter=10
    )
    normal_style = styles["BodyText"]

    #title
    content.append(
        Paragraph(
            "Qyronix AI Medical Report",
            title_style
        )
    )
    content.append(
        Paragraph(
            "Federated Learning Based Disease Detection System",
            styles["Italic"]
        )
    )
    content.append(Spacer(1,20))
    content.append(HRFlowable(width="100%"))
    content.append(Spacer(1,20))

    #patient table
    data = [
        ["Patient Name", last_result["name"]],
        ["Age", str(last_result["age"])],
        ["Disease Type", last_result["disease"]],
        ["Prediction", last_result["label"]],
        ["Confidence", f"{last_result['conf']:.2f}%"],
        ["Generated On",
         datetime.now().strftime("%d-%m-%Y %H:%M")] #current time of report generation
    ]

    table = Table(
        data,
        colWidths=[170,300]
    )
    #styling table
    table.setStyle(
        TableStyle([
            ("BACKGROUND",(0,0),(0,-1),
             colors.HexColor("#1e3a8a")),

            ("TEXTCOLOR",(0,0),(0,-1),
             colors.white),

            ("BACKGROUND",(1,0),(1,-1),
             colors.HexColor("#eff6ff")),

            ("GRID",(0,0),(-1,-1),1,
             colors.grey),

            ("FONTNAME",(0,0),(-1,-1),
             "Helvetica-Bold"),

            ("BOTTOMPADDING",(0,0),(-1,-1),10),

        ])
    )

    content.append(table)
    content.append(Spacer(1,25))

    #AI Analysis
    content.append(
        Paragraph(
            "AI Analysis",
            heading_style
        )
    )

    analysis = f"""
    Qyronix AI analyzed the uploaded medical image
    using deep learning and Explainable AI techniques.
    The system predicted <b>{last_result['label']}</b>
    with a confidence score of
    <b>{last_result['conf']:.2f}%</b>.
    """

    content.append(
        Paragraph(
            analysis,
            normal_style
        )
    )

    content.append(Spacer(1,20))

    #original image
    content.append(
        Paragraph(
            "Uploaded Image",
            heading_style
        )
    )

    if os.path.exists(last_result["img"]):
        img = RLImage(
            last_result["img"],
            width=250,
            height=250
        )
        img.hAlign = "CENTER"
        content.append(img)

    content.append(Spacer(1,25))

    #gradcam
    content.append(
        Paragraph(
            "Explainable AI Heatmap",
            heading_style
        )
    )
    if os.path.exists(last_result["cam"]):
        cam = RLImage(
            last_result["cam"],
            width=250,
            height=250
        )
        cam.hAlign = "CENTER"
        content.append(cam)

    content.append(Spacer(1,25))

    #footer
    footer = """
    <b>Disclaimer:</b>
    This AI-generated report is for educational
    and assistance purposes only.
    Please consult certified medical professionals
    before making healthcare decisions.
    """

    content.append(HRFlowable(width="100%"))
    content.append(Spacer(1,10))
    content.append(
        Paragraph(
            footer,
            styles["Italic"]
        )
    )
    #building pdf
    doc.build(content)
    return file_path


#home
@app.route("/", methods=["GET","POST"])
def home():
    try:
        with open("accuracy.json") as f:
            acc = json.load(f)

    except:
        acc = []

    if request.method == "POST":
        file = request.files["file"]
        filename = secure_filename(
            file.filename
        )
        os.makedirs(
            "static",
            exist_ok=True
        )
        path = os.path.join(
            "static",
            filename
        )

        file.save(path)
        disease = request.form["disease"]
        location_name = request.form[
            "location"
        ]
        user_lat, user_lon = get_coordinates(
            location_name
        )
        if user_lat is None:
            return render_template(
                "index.html",
                error="Location not found",
                acc=acc
            )

        label, conf, model = predict(
            path,
            disease
        )
        cam_path = generate_gradcam(
            path,
            model
        )
        hospitals = get_nearest_hospitals(
            disease,
            user_lat,
            user_lon
        )

        global last_result

        last_result = {
            "label": label,
            "conf": conf,
            "img": path,
            "cam": cam_path,
            "name": request.form["name"],
            "age": request.form["age"],
            "disease": disease
        }

        return render_template(
            "index.html",
            label=label,
            conf=conf,
            disease=disease,
            img=os.path.basename(path),
            cam=os.path.basename(cam_path),
            hospitals=hospitals,
            name=last_result["name"],
            age=last_result["age"],
            acc=acc
        )

    return render_template(
        "index.html",
        acc=acc
    )


#download

@app.route("/download")
def download():
    pdf = generate_pdf()
    if pdf is None:
        return "No report generated yet"
    return send_file(
        pdf,
        as_attachment=True
    )


#main
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )