from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Deepfake detection server is running"}), 200

@app.route('/process_image', methods=['POST'])
def process_image():
    return jsonify({
        "prediction": "real",
        "score": 0.95,
        "face_distortion": 0.05
    }), 200

@app.route('/process_video', methods=['POST'])
def process_video():
    return jsonify({
        "confidence": 0.90,
        "frame_consistency": 0.85,
        "audio_mismatch": 0.10,
        "risk_level": "low"
    }), 200

@app.route('/analyze_audio', methods=['POST'])
def analyze_audio():
    return jsonify({
        "cosine_similarity": 0.92,
        "mismatch_score": 0.08
    }), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
