# Digit Recognizer

A simple web app that uses a CNN to recognize handwritten digits (0–9) and sequences of digits. For my course assignment.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/AbdoJohanen/Number-Recognizer.git
   cd Number-Recognizer
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the App

1. Ensure the saved model file `mnist_cnn.keras` is in the project root.
2. Start the Flask server:

   ```bash
   python app.py
   ```
3. Open your browser at `http://127.0.0.1:5000`.

## Usage

* Use an existing image from /images or create your own a PNG or JPG image with black hand‑written digits on a white background.
* Upload the image.
* The app will preprocess the image, segment digits, and display:

  * The full predicted string (e.g. `381`).
  * Each 28×28 preprocessed digit image.

Enjoy testing your own handwritten numbers! Feel free to improve or extend.
