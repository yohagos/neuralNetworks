import os, numpy as np
from flask import Flask, render_template, request, redirect
from keras.api.models import load_model
from keras.src.utils import load_img, img_to_array
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads'

model = load_model('cat_vs_dog_model5820.keras')

CLASSES = ['Cat', 'Dog']

def preprocess_image(image_path):
    image = load_img(image_path, target_size=(150, 150))
    image = img_to_array(image)
    image /= 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_class(image, classes):
    prediction = model.predict(image)
    confidence = float(prediction[0][0])
    class_name = classes[int(confidence > 0.5)]
    return confidence * 100, class_name

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            processed_image = preprocess_image(filepath)
            confidence, class_name = predict_class(processed_image, CLASSES)
        except Exception as e:
            return f'Fehler bei Vorhersage: {e}'

        return render_template(
            'index.html',
            filename=filename,
            class_name=class_name,
            confidence=confidence
        )
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
