import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from feat.detector import Detector
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


UPLOAD_FOLDER = "./images"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

detector = Detector(
    emotion_model="resmasknet",
)
print(detector)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return "Home"


def get_max_emotion(emotion_result):
    # Get the emotion with the highest weight for each row
    max_emotions = emotion_result.idxmax(axis=1)
    # Get the counts of each emotion
    emotion_counts = max_emotions.value_counts()
    # Get the emotion with the highest count
    max_emotion = emotion_counts.idxmax()
    return max_emotion

@app.route('/media/upload', methods=['POST'])
def upload_media():
    if 'file' not in request.files:
        return jsonify({'error': 'media not provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        single_face_prediction = detector.detect_image(f'./images/{filename}')
        print(single_face_prediction.emotions)
        max_emotions = get_max_emotion(single_face_prediction.emotions)
        file_path = f'./images/{filename}'
        os.system(f'rm {file_path}')
    return jsonify({'msg': max_emotions})


if __name__ == "__main__":
    app.run(debug=True)
