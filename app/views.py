from flask import Blueprint, request, render_template
from .ans import predict

# initialize the main blueprint
main = Blueprint('main', __name__)

#GET, POST and definition of the one big function that is in use to handle all backend parts
@main.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        print(request.files)
        if 'image_file' not in request.files:
            print('no file uploaded')

        file = request.files['image_file']
        image = file.read()
        label = predict(image_bytes=image)

        return render_template('results.html', label=label)
