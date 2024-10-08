import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from typing import List
import ffmpeg
from flask import Flask, render_template, request

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('static','videos',f'{file_name}.mpg')
    # alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    # alignments = load_alignments(alignment_path)
    alignments = []
    return frames, alignments

def load_video(path:str) -> List[float]: 
    print(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

model = Sequential()
model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1,2,2)))

model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
model.add(Dropout(.5))

model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

model.reset_states()

model.load_weights('models_100_e/checkpoint')

def predict_text(file_path):
    sample = load_data(tf.convert_to_tensor(file_path))
    ### Compile the model
    yhat = model.predict(tf.expand_dims(sample[0], axis=0))
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
    e100 = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded][0].numpy().decode('utf-8')
    print("Epoch 100 : ", e100)
    return e100

# -------------
app = Flask(__name__)

# Define the directory for uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to load the deep learning model (Replace this with your model loading code)
def load_model():
    return

# Function to process uploaded video and generate text
def process_video(video_path):
    return

@app.route('/')
def index():
    # print(predict_text('.\\uploads\\bbaf3s.mpg'))
    # print(predict_text('.\\static\\videos\\s2_pwaq2a.mpg'))
    return render_template('index.html',video_name='', text_output="")

ALLOWED_EXTENSIONS = ['mpg']
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return "No video file found"
    video = request.files['video']
    if video.filename == '':
        return "No video file selected"
    if video and allowed_file(video.filename): 
        video.save('static/videos/'+video.filename)
        ffmpeg.input('static/videos/'+video.filename).output('static/videos/video.mp4',y='-y').run()
        return render_template('preview.html',video_name=video.filename, text_output=predict_text('.\\static\\videos\\'+video.filename))
    return "invalid file type"

# Main Function
if __name__ == '__main__':
    app.run(debug=True)
