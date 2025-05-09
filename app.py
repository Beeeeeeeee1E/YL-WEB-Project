from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import math
from io import BytesIO
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Создаем папку для загрузок, если ее нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class WebPhotoEditor:
    def __init__(self):
        self.current_image = None
        self.original_image = None
        self.temp_image = None
        self.selection = None  # (x, y, w, h)

    def load_image(self, file_path):
        self.original_image = Image.open(file_path)
        self.current_image = self.original_image.copy()
        self.temp_image = self.original_image.copy()
        return self.current_image

    def apply_filter(self, filter_name):
        if filter_name == 'gray':
            self.current_image = self.temp_image.convert('L').convert('RGB')
        elif filter_name == 'better_gray':
            img = self.temp_image.filter(ImageFilter.SMOOTH)
            self.current_image = img.convert('L').convert('RGB')
        elif filter_name == 'barilef':
            self.current_image = self.temp_image.filter(ImageFilter.EMBOSS)
        elif filter_name == 'blur':
            if self.selection:
                x, y, w, h = self.selection
                cropped = self.temp_image.crop((x, y, x + w, y + h))
                blurred = cropped.filter(ImageFilter.BLUR)
                self.current_image = self.temp_image.copy()
                self.current_image.paste(blurred, (x, y, x + w, y + h))
            else:
                self.current_image = self.temp_image.filter(ImageFilter.BLUR)
        return self.current_image

    def adjust_image(self, setting, value):
        if setting == 'contrast':
            enhancer = ImageEnhance.Contrast(self.temp_image)
            self.current_image = enhancer.enhance(float(value) / 100)
        elif setting == 'brightness':
            enhancer = ImageEnhance.Brightness(self.temp_image)
            self.current_image = enhancer.enhance(float(value) / 100)
        elif setting == 'sharpness':
            enhancer = ImageEnhance.Sharpness(self.temp_image)
            self.current_image = enhancer.enhance(float(value) / 100)
        elif setting == 'saturation':
            enhancer = ImageEnhance.Color(self.temp_image)
            self.current_image = enhancer.enhance(float(value) / 100)
        elif setting == 'transparency':
            img = self.temp_image.convert('RGBA')
            alpha = img.split()[3]
            alpha = Image.fromarray(np.array(alpha) * int(value) // 255)
            img.putalpha(alpha)
            self.current_image = img
        elif setting == 'color':
            r, g, b = self.temp_image.split()
            r = r.point(lambda i: int(255 * math.pow(float(i) / 255, int(value) / 50)))
            g = g.point(lambda i: int(255 * math.pow(float(i) / 255, int(value) / 50)))
            b = b.point(lambda i: int(255 * math.pow(float(i) / 255, int(value) / 50)))
            self.current_image = Image.merge('RGB', (r, g, b))
        return self.current_image

    def crop_image(self):
        if self.selection:
            x, y, w, h = self.selection
            self.current_image = self.temp_image.crop((x, y, x + w, y + h))
            self.temp_image = self.current_image.copy()
            self.selection = None
        return self.current_image

    def save_image(self, file_path):
        self.current_image.save(file_path)

    def set_selection(self, x, y, w, h):
        self.selection = (x, y, w, h)


editor = WebPhotoEditor()


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


@app.route('/')
def index():
    return render_template('editor.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        editor.load_image(filepath)
        img_base64 = image_to_base64(editor.current_image)
        return jsonify({
            'image': img_base64,
            'width': editor.current_image.width,
            'height': editor.current_image.height
        })

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/filter/<filter_name>', methods=['POST'])
def apply_filter(filter_name):
    if editor.current_image is None:
        return jsonify({'error': 'No image loaded'}), 400

    editor.apply_filter(filter_name)
    img_base64 = image_to_base64(editor.current_image)
    return jsonify({'image': img_base64})


@app.route('/adjust/<setting>', methods=['POST'])
def adjust_image(setting):
    if editor.current_image is None:
        return jsonify({'error': 'No image loaded'}), 400

    value = request.json.get('value')
    editor.adjust_image(setting, value)
    img_base64 = image_to_base64(editor.current_image)
    return jsonify({'image': img_base64})


@app.route('/crop', methods=['POST'])
def crop_image():
    if editor.current_image is None:
        return jsonify({'error': 'No image loaded'}), 400

    editor.crop_image()
    img_base64 = image_to_base64(editor.current_image)
    return jsonify({
        'image': img_base64,
        'width': editor.current_image.width,
        'height': editor.current_image.height
    })


@app.route('/selection', methods=['POST'])
def set_selection():
    if editor.current_image is None:
        return jsonify({'error': 'No image loaded'}), 400

    x = request.json.get('x')
    y = request.json.get('y')
    w = request.json.get('w')
    h = request.json.get('h')

    editor.set_selection(x, y, w, h)
    return jsonify({'success': True})


@app.route('/save', methods=['POST'])
def save_image():
    if editor.current_image is None:
        return jsonify({'error': 'No image loaded'}), 400

    filename = request.json.get('filename', 'edited_image.png')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    editor.save_image(filepath)
    return jsonify({'url': url_for('download_file', filename=filename)})


@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
