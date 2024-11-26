from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_pymongo import PyMongo
import random
import torch
from scipy.io.wavfile import read
import whisper
from speechbrain.pretrained import SpeakerRecognition
from io import BytesIO
from pydub import AudioSegment
import numpy as np
import tempfile
import uuid  # For unique recording IDs
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
app.secret_key = 'abcdefghi'  # Replace with a strong, unpredictable secret key
app.secret_key = os.urandom(24)

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/voice_authentication_system"
mongo = PyMongo(app)
users_collection = mongo.db.users

# Initialize Whisper model for speech-to-text
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

# Initialize SpeakerRecognition model from Speechbrain
print("Loading SpeakerRecognition model...")
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")
print("SpeakerRecognition model loaded.")

# List of 200 questions for enrollment
questions = [
    "What is your favorite color?",
    "What is your pet's name?",
    "What city were you born in?",
    "What is your favorite food?",
    "What is the name of your first school?",
    "What hobby do you enjoy the most?",
    "What is your favorite movie?",
    "What is your favorite book?",
    "What is your favorite sport?",
    "What is your favorite season?",
    # ... Add more unique questions up to 200
] + [f"Sample question {i}" for i in range(11, 201)]  # Filling up to 200 questions

def convert_audio_to_wav(audio_data):
    """Convert uploaded audio data to WAV format using pydub."""
    try:
        audio = AudioSegment.from_file(BytesIO(audio_data))
        byte_io = BytesIO()
        audio.export(byte_io, format="wav")
        byte_io.seek(0)
        wav_data = byte_io.read()
        return wav_data
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None



def compute_embedding(audio_data):
    """Generate an embedding from the audio data using Speechbrain."""
    byte_io = BytesIO(audio_data)
    try:
        sample_rate, audio_array = read(byte_io)
    except Exception as e:
        print(f"Error reading WAV data: {e}")
        return None

    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    # Resample to 16000 Hz if necessary
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        num_samples = int(len(audio_array) * target_sample_rate / sample_rate)
        audio_array = np.interp(
            np.linspace(0.0, 1.0, num_samples, endpoint=False),
            np.linspace(0.0, 1.0, len(audio_array)),
            audio_array
        )

    # Normalize audio to float32
    if np.max(np.abs(audio_array)) > 0:
        audio_array = audio_array.astype(np.float32) / np.max(np.abs(audio_array))
    else:
        audio_array = audio_array.astype(np.float32)

    # Convert to torch tensor
    audio_tensor = torch.tensor(audio_array).unsqueeze(0)

    # Generate embedding
    embedding = verification.encode_batch(audio_tensor)

    # Squeeze to remove extra dimensions
    embedding = embedding.squeeze(0)

    return embedding


def transcribe_audio(audio_data):
    """Transcribe the audio data using Whisper."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            tmpfile.write(audio_data)
            tmpfile.flush()
            result = whisper_model.transcribe(tmpfile.name)
        return result['text']
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return ""

def compute_cosine_similarity(new_embedding, stored_embeddings):
    """Compute cosine similarity between new embedding and all stored embeddings."""
    similarities = []
    new_emb = np.array(new_embedding)
    
    # Ensure new_emb is 1D
    if new_emb.ndim != 1:
        new_emb = new_emb.flatten()
    
    # Reshape to 2D array as expected by cosine_similarity
    new_emb = new_emb.reshape(1, -1)  # Shape: (1, n_features)
    
    for emb in stored_embeddings:
        stored_emb = np.array(emb)
        
        # Ensure stored_emb is 1D
        if stored_emb.ndim != 1:
            stored_emb = stored_emb.flatten()
        
        # Reshape to 2D array
        stored_emb = stored_emb.reshape(1, -1)  # Shape: (1, n_features)
        
        similarity = cosine_similarity(new_emb, stored_emb)[0][0]
        similarities.append(similarity)
    
    return similarities

@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/authenticate', methods=['POST'])
def authenticate():
    user_id = request.form['user_id'].strip()

    if not user_id:
        return render_template('login.html', message="Username cannot be empty.")

    # Check if user exists
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return render_template('login.html', message="Username not found. Please enroll first.")

    # Check if user has any questions
    if not user.get('answers'):
        return render_template('login.html', message="No enrolled questions found. Please enroll first.")

    # Select one random question from the enrolled questions
    enrolled_questions = list(user['answers'].keys())
    selected_question = random.choice(enrolled_questions)

    # Store the selected question in session
    session['user_id'] = user_id
    session['selected_question'] = selected_question

    return render_template('verify.html', question=selected_question)

@app.route('/secure_dashboard')
def secure_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('secure_dashboard.html')


@app.route('/verify_answer', methods=['POST'])
def verify_answer():
    if 'user_id' not in session or 'selected_question' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    question = session['selected_question']
    audio_file = request.files['audio']

    if not audio_file:
        return jsonify({"status": "error", "message": "No audio file provided."}), 400

    audio_data = audio_file.read()

    # Convert audio to WAV
    wav_data = convert_audio_to_wav(audio_data)
    if not wav_data:
        return jsonify({"status": "error", "message": "Audio conversion failed."}), 500

    # Transcribe audio
    transcription = transcribe_audio(wav_data)

    if transcription.strip() == "":
        return jsonify({"status": "error", "message": "Transcription failed or was empty."}), 500

    # Compute embedding
    embedding = compute_embedding(wav_data)
    if embedding is None:
        return jsonify({"status": "error", "message": "Embedding computation failed."}), 500

    # Retrieve all stored embeddings for the user across all questions
    user = users_collection.find_one({"user_id": user_id})
    stored_embeddings = []
    for q_data in user.get('answers', {}).values():
        for emb in q_data.get('embeddings', []):
            stored_embeddings.append(emb['embedding'])

    if not stored_embeddings:
        return jsonify({"status": "error", "message": "No stored embeddings found for comparison."}), 500

    # Compute cosine similarities
    similarities = compute_cosine_similarity(embedding.tolist(), stored_embeddings)

    # Define similarity threshold
    THRESHOLD = 0.5  # Adjust based on your requirements

    # Check if any similarity exceeds the threshold
    max_similarity = max(similarities)
    if max_similarity >= THRESHOLD:
        result = "open"
    else:
        result = "access denied"

    return jsonify({"status": "success", "result": result, "transcription": transcription.lower()})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/enroll', methods=['POST'])
def enroll():
    user_id = request.form['user_id'].strip()

    if not user_id:
        return render_template('index.html', message="Username cannot be empty.")

    # Check if user already exists
    if users_collection.find_one({"user_id": user_id}):
        return render_template('index.html', message="Username already exists. Please choose a different one.")

    # Initialize user data with empty answers (dictionary with questions as keys and their data)
    users_collection.insert_one({"user_id": user_id, "answers": {}})

    return render_template('enroll.html', user_id=user_id)

@app.route('/get_questions', methods=['GET'])
def get_questions():
    selected_questions = random.sample(questions, 3)
    return jsonify({"questions": selected_questions})

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    user_id = request.form['user_id']
    question = request.form['question']
    audio_file = request.files['audio']

    if not audio_file:
        return jsonify({"status": "error", "message": "No audio file provided."}), 400

    audio_data = audio_file.read()

    # Convert audio to WAV
    wav_data = convert_audio_to_wav(audio_data)
    if not wav_data:
        return jsonify({"status": "error", "message": "Audio conversion failed."}), 500

    # Transcribe audio
    transcription = transcribe_audio(wav_data)

    if transcription.strip() == "":
        return jsonify({"status": "error", "message": "Transcription failed or was empty."}), 500

    # Compute embedding
    embedding = compute_embedding(wav_data)
    if embedding is None:
        return jsonify({"status": "error", "message": "Embedding computation failed."}), 500

    # Convert embedding to list for JSON serialization
    embedding_list = embedding.tolist()

    # Generate a unique ID for the recording
    recording_id = str(uuid.uuid4())

    # Prepare the data to store
    embedding_data = {
        "recording_id": recording_id,
        "embedding": embedding_list
    }

    # Insert or update user data
    user = users_collection.find_one({"user_id": user_id})
    if question not in user.get('answers', {}):
        users_collection.update_one(
            {"user_id": user_id},
            {"$set": {f"answers.{question}.question_text": question, f"answers.{question}.transcription": transcription, f"answers.{question}.embeddings": []}},
            upsert=True
        )
    else:
        # Update the transcription to the latest one
        users_collection.update_one(
            {"user_id": user_id},
            {"$set": {f"answers.{question}.transcription": transcription}}
        )

    # Retrieve current embeddings for the question
    user = users_collection.find_one({"user_id": user_id})
    current_embeddings = user.get('answers', {}).get(question, {}).get('embeddings', [])

    if len(current_embeddings) >= 3:
        return jsonify({"status": "error", "message": "Maximum recording attempts reached for this question."}), 400

    # Append the new embedding
    users_collection.update_one(
        {"user_id": user_id},
        {"$push": {f"answers.{question}.embeddings": embedding_data}}
    )

    # Check if the user has answered all three questions
    user = users_collection.find_one({"user_id": user_id})
    if len(user.get('answers', {})) >= 3:
        return jsonify({"status": "complete", "message": "Enrollment complete."})

    return jsonify({"status": "success", "transcription": transcription, "recording_id": recording_id})

@app.route('/delete_recording', methods=['POST'])
def delete_recording():
    user_id = request.form['user_id']
    question = request.form['question']
    recording_id = request.form['recording_id']

    # Remove the embedding from the user's data
    result = users_collection.update_one(
        {"user_id": user_id},
        {"$pull": {f"answers.{question}.embeddings": {"recording_id": recording_id}}}
    )

    if result.modified_count > 0:
        # Update transcription if embeddings remain
        user = users_collection.find_one({"user_id": user_id})
        embeddings = user.get('answers', {}).get(question, {}).get('embeddings', [])

        if embeddings:
            # Optionally, update the transcription to the latest one
            latest_transcription = user.get('answers', {}).get(question, {}).get('transcription', "")
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {f"answers.{question}.transcription": latest_transcription}}
            )
        else:
            # If no embeddings left, remove the question from answers
            users_collection.update_one(
                {"user_id": user_id},
                {"$unset": {f"answers.{question}": ""}}
            )

        return jsonify({"status": "success", "message": "Recording deleted."})
    else:
        return jsonify({"status": "error", "message": "Recording not found."}), 404

@app.route('/complete/<user_id>')
def complete(user_id):
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return "User not found", 404
    return render_template('complete.html', user=user)

@app.route('/cancel_enrollment', methods=['POST'])
def cancel_enrollment():
    user_id = request.form['user_id']
    users_collection.delete_one({"user_id": user_id})
    return jsonify({"status": "success", "message": "Enrollment canceled and data deleted."})

if __name__ == '__main__':
    app.run(debug=True)
