import os
import io
import base64

import cv2
import numpy as np
import tensorflow as tf

from flask import Flask, request, render_template
from PIL import Image, ImageOps

app = Flask(__name__)
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'mnist_cnn.keras')
model = tf.keras.models.load_model(model_path)


def segment_and_predict(pil_img):
    """
    Tar en PIL-bild med flera handskrivna siffror,
    segmenterar via vertikal histogrammetod,
    förbehandlar varje tecken och predikterar.
    Returnerar:
      preds:   lista med str(pred) för varje segment
      img_uris: lista med base64-URL för varje segmentbild (28×28)
    """

    # 1) Gråskala + invertera (vit bakgrund → svart, svart streck → vitt)
    gray = pil_img.convert('L')
    inv = ImageOps.invert(gray)
    np_img = np.array(inv)

    # 2) Tröskla till binärbild
    _, thresh = cv2.threshold(np_img, 128, 255, cv2.THRESH_BINARY)

    # 3) Vertikal projektion för att hitta kolumner där det finns pixlar
    mask = np.sum(thresh > 0, axis=0) > 0  # bool-array över bildens bredd
    segments = []
    in_seg = False
    for x, val in enumerate(mask):
        if val and not in_seg:
            start = x
            in_seg = True
        elif not val and in_seg:
            end = x
            in_seg = False
            segments.append((start, end))
    if in_seg:
        segments.append((start, len(mask)))

    preds = []
    img_uris = []

    # 4) Loop över alla segmentkolumner
    for (x1, x2) in segments:
        # 4a) Beskär horisontellt
        region = thresh[:, x1:x2]
        # 4b) Vertikal beskärning: hitta rader med pixlar
        rows = np.sum(region > 0, axis=1)
        ys = np.where(rows > 0)[0]
        if ys.size == 0:
            continue
        y1, y2 = ys[0], ys[-1] + 1
        digit = region[y1:y2, :]

        # 5) Proportionellt skalad till max 20px
        h, w = digit.shape
        if w > h:
            new_w = 20
            new_h = max(1, int(20 * (h / w)))
        else:
            new_h = 20
            new_w = max(1, int(20 * (w / h)))
        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 6) Padding till 28×28 och centrera
        pad_x = (28 - new_w) // 2
        pad_y = (28 - new_h) // 2
        padded = np.pad(
            digit,
            ((pad_y, 28 - new_h - pad_y), (pad_x, 28 - new_w - pad_x)),
            mode='constant',
            constant_values=0
        )

        # 7) Gör om till PIL för att base64-koda
        pil_seg = Image.fromarray(padded)
        buf = io.BytesIO()
        pil_seg.save(buf, format='PNG')
        buf.seek(0)
        uri = base64.b64encode(buf.read()).decode('utf-8')
        img_uris.append(f"data:image/png;base64,{uri}")

        # 8) Normalisera + reshape för modellen
        arr = padded.astype('float32') / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        # 9) Prediktion
        prediction = model.predict(arr).argmax()
        preds.append(str(int(prediction)))

    return preds, img_uris


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        f = request.files['file']
        pil_img = Image.open(f)

        preds, uris = segment_and_predict(pil_img)
        full_pred = ''.join(preds) if preds else "Ingen siffra hittad"

        return render_template(
            'result.html',
            full_prediction=full_pred,
            segment_images=uris
        )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
