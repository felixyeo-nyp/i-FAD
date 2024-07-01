import datetime
import shelve
import threading
import time
import re
from datetime import timedelta, datetime

import cv2
import torch
import torchvision
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for, send_from_directory
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from flask_cors import CORS, cross_origin

from Forms import configurationForm, emailForm
from OOP import Line_Chart_Data

object_count = {1: 0}

def create_app():
    app = Flask(__name__)
    CORS(app)
    # Configuration, routes, extensions, etc.
    app.config['SERVER_NAME'] = None

    # Check GPU availability
    if torch.cuda.is_available():
        print('Using GPU for video processing')
    else:
        print('No GPU found, using CPU for video processing')

    # Other configurations and route definitions can be added here

    return app

# Define class for real-time video processing
class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()
        self.condition = threading.Condition()
        self.is_running = False
        self.frame = None
        self.pellets_num = 0
        self.callback = None
        super().__init__(name=name)
        self.start()

    def start(self):
        self.is_running = True
        super().start()

    def stop(self, timeout=None):
        self.is_running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.is_running:
            (rv, img) = self.capture.read()
            assert rv
            counter += 1
            with self.condition:
                self.frame = img if rv else None
                self.condition.notify_all()
            if self.callback:
                self.callback(img)

    def read(self, wait=True, sequence_number=None, timeout=None):
        with self.condition:
            if wait:
                if sequence_number is None:
                    sequence_number = self.pellets_num + 1
                if sequence_number < 1:
                    sequence_number = 1
                if sequence_number > 0:
                    self.pellets_num = sequence_number
                rv = self.condition.wait_for(lambda: self.pellets_num >= sequence_number, timeout=timeout)
                if not rv:
                    return (self.pellets_num, self.frame)
            return (self.pellets_num, self.frame)

# Define labels and model paths
class_labels = {
    1: 'Pellets',
    2: 'Fecal Matters'
}
model_path = './best_model.pth'

# Function to create a Faster R-CNN model
def create_model(num_classes, pretrained=False, coco_model=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    if not coco_model:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Function to load the Faster R-CNN model
def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
    return model

# Function to generate video frames for streaming
def generate_frames():
    cap = cv2.VideoCapture('./static/testing.mp4')
    fresh = FreshestFrame(cap)

    # Load the Faster R-CNN model
    num_classes = 2  # Assuming 2 classes for 'Pellets' and background
    model = load_model(model_path, num_classes)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    while True:
        temp_object_count = {1: 0}
        cnt, frame = fresh.read(sequence_number=temp_object_count[1] + 1)
        if frame is None:
            break

        img_tensor = torchvision.transforms.ToTensor()(frame).to(device)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            predictions = model(img_tensor)

        for i in range(len(predictions[0]['labels'])):
            label = predictions[0]['labels'][i].item()
            if label in class_labels and label == 1:
                box = predictions[0]['boxes'][i].cpu().numpy().astype(int)
                score = predictions[0]['scores'][i].item()
                if label == 1 and score > 0.5:  # Adjust score threshold as needed
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_labels[label]}: {score:.2f}', (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    temp_object_count[label] += 1

        for label, count in temp_object_count.items():
            if label == 1:
                object_count[label] = count

        for label, count in object_count.items():
            text = f'{class_labels[label]} Count: {count}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_position = (frame.shape[1] - text_size[0] - 10, 30 * (label + 1))
            cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    fresh.stop()
    cap.release()

app = create_app()

# Flask routes for dashboard, health check, pellet counts, settings, and email settings
@app.route('/', methods=['GET', 'POST'])
def dashboard():
    edit_form = configurationForm(request.form)
    db = shelve.open('settings.db', 'r')
    Time_Record_dict = db['Time_Record']
    db.close()
    id_array = []
    for key in Time_Record_dict:
        product = Time_Record_dict.get(key)
        if key == "Time_Record_Info":
            id_array.append(product)
    return render_template('dashboard.html', count=len(id_array), id_array=id_array, edit=0, form=edit_form)

@app.route('/healthz')
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/pellet_counts')
def pellet_counts():
    global object_count
    timestamps = [time.strftime('%H:%M:%S') for _ in object_count]
    counts = list(object_count.values())
    data = {
        'labels': timestamps,
        'data': counts
    }
    return jsonify(data)

@app.route('/update', methods=['GET', 'POST'])
def update_setting():
    setting = configurationForm(request.form)

    if request.method == 'POST' and setting.validate():
        pattern = r'^([01]\d|2[0-3]):([0-5]\d)$'

        if re.match(pattern, setting.first_timer.data) and re.match(pattern, setting.second_timer.data):
            first_hour = int(setting.first_timer.data.split(':')[0])
            second_hour = int(setting.second_timer.data.split(':')[0])
            if (6 <= first_hour <= 12) and (12 <= second_hour <= 24):
                db = shelve.open('settings.db', 'w')
                Time_Record_dict = db['Time_Record']
                j = Time_Record_dict.get('Time_Record_Info')
                j.set_first_timer(setting.first_timer.data)
                j.set_second_timer(setting.second_timer.data)
                j.set_pellets(setting.pellets.data)
                j.set_seconds(setting.seconds.data)
                j.set_confidence(setting.confidence.data)
                db['Time_Record'] = Time_Record_dict
                db.close()
                return redirect(url_for('dashboard'))
            elif not (6 <= first_hour <= 12):
                setting.first_timer.errors.append('First timer should be between 06:00 and 12:00 (morning to afternoon).')
                return render_template('settings.html', form=setting)
            else:
                setting.second_timer.errors.append('Second timer should be between 12:00 and 24:00 (afternoon to night).')
                return render_template('settings.html', form=setting)
        elif not re.match(pattern, setting.first_timer.data):
            setting.first_timer.errors.append('Invalid time format. Please use HH:MM format.')
            return render_template('settings.html', form=setting)
        elif not re.match(pattern, setting.second_timer.data):
            setting.second_timer.errors.append('Invalid time format. Please use HH:MM format.')
            return render_template('settings.html', form=setting)
    return render_template('settings.html', form=setting)

@app.route('/email_settings', methods=['GET', 'POST'])
def email_setting():
    email = emailForm(request.form)

    if request.method == 'POST' and email.validate():
        db = shelve.open('settings.db', 'w')
        Time_Record_dict = db['Time_Record']
        j = Time_Record_dict.get('Time_Record_Info')
        j.set_recipient_email(email.email.data)
        j.set_sender_email(email.email.data)
        j.set_email_password(email.password.data)
        db['Time_Record'] = Time_Record_dict
        db.close()
        return redirect(url_for('dashboard'))
    return render_template('email.html', form=email)

@app.route('/video_feed')
@cross_origin()
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
