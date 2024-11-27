# from flask import Flask, render_template, request, jsonify, redirect, url_for
# from flask_pymongo import PyMongo
# import random
# import torch
# from scipy.io.wavfile import read
# import whisper
# from speechbrain.pretrained import SpeakerRecognition
# from io import BytesIO
# from pydub import AudioSegment
# import numpy as np
# import tempfile
# import uuid  # For unique recording IDs
# from flask import Flask, render_template, request, jsonify, redirect, url_for, session
# import os
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# app = Flask(__name__)
# app.secret_key = 'abcdefghi'  # Replace with a strong, unpredictable secret key
# app.secret_key = os.urandom(24)

# # MongoDB configuration
# app.config["MONGO_URI"] = "mongodb://localhost:27017/voice_authentication_system"
# mongo = PyMongo(app)
# users_collection = mongo.db.users

# # Initialize Whisper model for speech-to-text
# print("Loading Whisper model...")
# whisper_model = whisper.load_model("base")
# print("Whisper model loaded.")

# # Initialize SpeakerRecognition model from Speechbrain
# print("Loading SpeakerRecognition model...")
# verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")
# print("SpeakerRecognition model loaded.")

# # List of 200 questions for enrollment
# questions = [
#     "What is your favorite color?",
#     "What is your pet's name?",
#     "What city were you born in?",
#     "What is your favorite food?",
#     "What is the name of your first school?",
#     "What hobby do you enjoy the most?",
#     "What is your favorite movie?",
#     "What is your favorite book?",
#     "What is your favorite sport?",
#     "What is your favorite season?",
#     # ... Add more unique questions up to 200
# ] + [f"Sample question {i}" for i in range(11, 201)]  # Filling up to 200 questions

# def convert_audio_to_wav(audio_data):
#     """Convert uploaded audio data to WAV format using pydub."""
#     try:
#         audio = AudioSegment.from_file(BytesIO(audio_data))
#         byte_io = BytesIO()
#         audio.export(byte_io, format="wav")
#         byte_io.seek(0)
#         wav_data = byte_io.read()
#         return wav_data
#     except Exception as e:
#         print(f"Error converting audio: {e}")
#         return None



# def compute_embedding(audio_data):
#     """Generate an embedding from the audio data using Speechbrain."""
#     byte_io = BytesIO(audio_data)
#     try:
#         sample_rate, audio_array = read(byte_io)
#     except Exception as e:
#         print(f"Error reading WAV data: {e}")
#         return None

#     if audio_array.ndim > 1:
#         audio_array = audio_array[:, 0]

#     # Resample to 16000 Hz if necessary
#     target_sample_rate = 16000
#     if sample_rate != target_sample_rate:
#         num_samples = int(len(audio_array) * target_sample_rate / sample_rate)
#         audio_array = np.interp(
#             np.linspace(0.0, 1.0, num_samples, endpoint=False),
#             np.linspace(0.0, 1.0, len(audio_array)),
#             audio_array
#         )

#     # Normalize audio to float32
#     if np.max(np.abs(audio_array)) > 0:
#         audio_array = audio_array.astype(np.float32) / np.max(np.abs(audio_array))
#     else:
#         audio_array = audio_array.astype(np.float32)

#     # Convert to torch tensor
#     audio_tensor = torch.tensor(audio_array).unsqueeze(0)

#     # Generate embedding
#     embedding = verification.encode_batch(audio_tensor)

#     # Squeeze to remove extra dimensions
#     embedding = embedding.squeeze(0)

#     return embedding


# def transcribe_audio(audio_data):
#     """Transcribe the audio data using Whisper."""
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
#             tmpfile.write(audio_data)
#             tmpfile.flush()
#             result = whisper_model.transcribe(tmpfile.name)
#         return result['text']
#     except Exception as e:
#         print(f"Error transcribing audio: {e}")
#         return ""

# def compute_cosine_similarity(new_embedding, stored_embeddings):
#     """Compute cosine similarity between new embedding and all stored embeddings."""
#     similarities = []
#     new_emb = np.array(new_embedding)
    
#     # Ensure new_emb is 1D
#     if new_emb.ndim != 1:
#         new_emb = new_emb.flatten()
    
#     # Reshape to 2D array as expected by cosine_similarity
#     new_emb = new_emb.reshape(1, -1)  # Shape: (1, n_features)
    
#     for emb in stored_embeddings:
#         stored_emb = np.array(emb)
        
#         # Ensure stored_emb is 1D
#         if stored_emb.ndim != 1:
#             stored_emb = stored_emb.flatten()
        
#         # Reshape to 2D array
#         stored_emb = stored_emb.reshape(1, -1)  # Shape: (1, n_features)
        
#         similarity = cosine_similarity(new_emb, stored_emb)[0][0]
#         similarities.append(similarity)
    
#     return similarities

# @app.route('/login')
# def login():
#     return render_template('login.html')


# @app.route('/authenticate', methods=['POST'])
# def authenticate():
#     user_id = request.form['user_id'].strip()

#     if not user_id:
#         return render_template('login.html', message="Username cannot be empty.")

#     # Check if user exists
#     user = users_collection.find_one({"user_id": user_id})
#     if not user:
#         return render_template('login.html', message="Username not found. Please enroll first.")

#     # Check if user has any questions
#     if not user.get('answers'):
#         return render_template('login.html', message="No enrolled questions found. Please enroll first.")

#     # Select one random question from the enrolled questions
#     enrolled_questions = list(user['answers'].keys())
#     selected_question = random.choice(enrolled_questions)

#     # Store the selected question in session
#     session['user_id'] = user_id
#     session['selected_question'] = selected_question

#     return render_template('verify.html', question=selected_question)

# @app.route('/secure_dashboard')
# def secure_dashboard():
#     if 'user_id' not in session:
#         return redirect(url_for('login'))
#     return render_template('secure_dashboard.html')


# @app.route('/verify_answer', methods=['POST'])
# def verify_answer():
#     if 'user_id' not in session or 'selected_question' not in session:
#         return redirect(url_for('login'))

#     user_id = session['user_id']
#     question = session['selected_question']
#     audio_file = request.files['audio']

#     if not audio_file:
#         return jsonify({"status": "error", "message": "No audio file provided."}), 400

#     audio_data = audio_file.read()

#     # Convert audio to WAV
#     wav_data = convert_audio_to_wav(audio_data)
#     if not wav_data:
#         return jsonify({"status": "error", "message": "Audio conversion failed."}), 500

#     # Transcribe audio
#     transcription = transcribe_audio(wav_data)

#     if transcription.strip() == "":
#         return jsonify({"status": "error", "message": "Transcription failed or was empty."}), 500

#     # Compute embedding
#     embedding = compute_embedding(wav_data)
#     if embedding is None:
#         return jsonify({"status": "error", "message": "Embedding computation failed."}), 500

#     # Retrieve all stored embeddings for the user across all questions
#     user = users_collection.find_one({"user_id": user_id})
#     stored_embeddings = []
#     for q_data in user.get('answers', {}).values():
#         for emb in q_data.get('embeddings', []):
#             stored_embeddings.append(emb['embedding'])

#     if not stored_embeddings:
#         return jsonify({"status": "error", "message": "No stored embeddings found for comparison."}), 500

#     # Compute cosine similarities
#     similarities = compute_cosine_similarity(embedding.tolist(), stored_embeddings)

#     # Define similarity threshold
#     THRESHOLD = 0.5  # Adjust based on your requirements

#     # Check if any similarity exceeds the threshold
#     max_similarity = max(similarities)
#     if max_similarity >= THRESHOLD:
#         result = "open"
#     else:
#         result = "access denied"

#     return jsonify({"status": "success", "result": result, "transcription": transcription.lower()})

# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('login'))


# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/enroll', methods=['POST'])
# def enroll():
#     user_id = request.form['user_id'].strip()

#     if not user_id:
#         return render_template('index.html', message="Username cannot be empty.")

#     # Check if user already exists
#     if users_collection.find_one({"user_id": user_id}):
#         return render_template('index.html', message="Username already exists. Please choose a different one.")

#     # Initialize user data with empty answers (dictionary with questions as keys and their data)
#     users_collection.insert_one({"user_id": user_id, "answers": {}})

#     return render_template('enroll.html', user_id=user_id)

# @app.route('/get_questions', methods=['GET'])
# def get_questions():
#     selected_questions = random.sample(questions, 3)
#     return jsonify({"questions": selected_questions})

# @app.route('/submit_answer', methods=['POST'])
# def submit_answer():
#     user_id = request.form['user_id']
#     question = request.form['question']
#     audio_file = request.files['audio']

#     if not audio_file:
#         return jsonify({"status": "error", "message": "No audio file provided."}), 400

#     audio_data = audio_file.read()

#     # Convert audio to WAV
#     wav_data = convert_audio_to_wav(audio_data)
#     if not wav_data:
#         return jsonify({"status": "error", "message": "Audio conversion failed."}), 500

#     # Transcribe audio
#     transcription = transcribe_audio(wav_data)

#     if transcription.strip() == "":
#         return jsonify({"status": "error", "message": "Transcription failed or was empty."}), 500

#     # Compute embedding
#     embedding = compute_embedding(wav_data)
#     if embedding is None:
#         return jsonify({"status": "error", "message": "Embedding computation failed."}), 500

#     # Convert embedding to list for JSON serialization
#     embedding_list = embedding.tolist()

#     # Generate a unique ID for the recording
#     recording_id = str(uuid.uuid4())

#     # Prepare the data to store
#     embedding_data = {
#         "recording_id": recording_id,
#         "embedding": embedding_list
#     }

#     # Insert or update user data
#     user = users_collection.find_one({"user_id": user_id})
#     if question not in user.get('answers', {}):
#         users_collection.update_one(
#             {"user_id": user_id},
#             {"$set": {f"answers.{question}.question_text": question, f"answers.{question}.transcription": transcription, f"answers.{question}.embeddings": []}},
#             upsert=True
#         )
#     else:
#         # Update the transcription to the latest one
#         users_collection.update_one(
#             {"user_id": user_id},
#             {"$set": {f"answers.{question}.transcription": transcription}}
#         )

#     # Retrieve current embeddings for the question
#     user = users_collection.find_one({"user_id": user_id})
#     current_embeddings = user.get('answers', {}).get(question, {}).get('embeddings', [])

#     if len(current_embeddings) >= 3:
#         return jsonify({"status": "error", "message": "Maximum recording attempts reached for this question."}), 400

#     # Append the new embedding
#     users_collection.update_one(
#         {"user_id": user_id},
#         {"$push": {f"answers.{question}.embeddings": embedding_data}}
#     )

#     # Check if the user has answered all three questions
#     user = users_collection.find_one({"user_id": user_id})
#     if len(user.get('answers', {})) >= 3:
#         return jsonify({"status": "complete", "message": "Enrollment complete."})

#     return jsonify({"status": "success", "transcription": transcription, "recording_id": recording_id})

# @app.route('/delete_recording', methods=['POST'])
# def delete_recording():
#     user_id = request.form['user_id']
#     question = request.form['question']
#     recording_id = request.form['recording_id']

#     # Remove the embedding from the user's data
#     result = users_collection.update_one(
#         {"user_id": user_id},
#         {"$pull": {f"answers.{question}.embeddings": {"recording_id": recording_id}}}
#     )

#     if result.modified_count > 0:
#         # Update transcription if embeddings remain
#         user = users_collection.find_one({"user_id": user_id})
#         embeddings = user.get('answers', {}).get(question, {}).get('embeddings', [])

#         if embeddings:
#             # Optionally, update the transcription to the latest one
#             latest_transcription = user.get('answers', {}).get(question, {}).get('transcription', "")
#             users_collection.update_one(
#                 {"user_id": user_id},
#                 {"$set": {f"answers.{question}.transcription": latest_transcription}}
#             )
#         else:
#             # If no embeddings left, remove the question from answers
#             users_collection.update_one(
#                 {"user_id": user_id},
#                 {"$unset": {f"answers.{question}": ""}}
#             )

#         return jsonify({"status": "success", "message": "Recording deleted."})
#     else:
#         return jsonify({"status": "error", "message": "Recording not found."}), 404

# @app.route('/complete/<user_id>')
# def complete(user_id):
#     user = users_collection.find_one({"user_id": user_id})
#     if not user:
#         return "User not found", 404
#     return render_template('complete.html', user=user)

# @app.route('/cancel_enrollment', methods=['POST'])
# def cancel_enrollment():
#     user_id = request.form['user_id']
#     users_collection.delete_one({"user_id": user_id})
#     return jsonify({"status": "success", "message": "Enrollment canceled and data deleted."})

# if __name__ == '__main__':
#     app.run(debug=True)




from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file
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
import uuid 
from werkzeug.utils import secure_filename
import os
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import glob
import time
import io
import warnings
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, send_file, flash
from flask import jsonify
from difflib import SequenceMatcher
# ###############################################################################################################################
# ###############################################################################################################################

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ###############################################################################################################################
# ###############################################################################################################################

warnings.filterwarnings("ignore", category=FutureWarning)
# ###############################################################################################################################
# ###############################################################################################################################


app = Flask(__name__)
app.secret_key = 'abcdefghi'  # Replace with a strong, unpredictable secret key
app.secret_key = os.urandom(24)
# ###############################################################################################################################
# ###############################################################################################################################

# MongoDB configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/voice_authentication_system"
mongo = PyMongo(app)
users_collection = mongo.db.users
# ###############################################################################################################################
# ###############################################################################################################################

# Initialize Whisper model for speech-to-text
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

# ###############################################################################################################################
# ###############################################################################################################################

# Initialize SpeakerRecognition model from Speechbrain
print("Loading SpeakerRecognition model...")
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")
print("SpeakerRecognition model loaded.")
# ###############################################################################################################################
# ###############################################################################################################################


unique_questions  = [
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
    "What is your mother's maiden name?",
    "What was the make of your first car?",
    "What was your childhood nickname?",
    "In what town did your parents meet?",
    "What was the name of your first pet?",
    "What is your favorite place to vacation?",
    "What was the first concert you attended?",
    "What is your favorite type of music?",
    "What is your favorite TV show?",
    "What was your favorite subject in school?",
    "What was the first company you worked for?",
    "What is your favorite dessert?",
    "What is your favorite holiday?",
    "What was your high school mascot?",
    "What was the model of your first phone?",
    "What was the name of your first roommate?",
    "What is your favorite restaurant?",
    "What is your favorite quote?",
    "What was your dream job as a child?",
    "What is the first name of your oldest sibling?",
    "What is your favorite animal?",
    "What is your favorite outdoor activity?",
    "What is your favorite indoor activity?",
    "What is your favorite board game?",
    "What is your favorite video game?",
    "What was the name of your elementary school?",
    "What was your favorite toy as a child?",
    "What is your favorite fruit?",
    "What is your favorite vegetable?",
    "What is your favorite ice cream flavor?",
    "What is your favorite candy?",
    "What was the name of your first boss?",
    "What is your favorite drink?",
    "What is your favorite type of cuisine?",
    "What is your favorite flower?",
    "What is your favorite type of weather?",
    "What is your favorite kind of art?",
    "What is your favorite store?",
    "What is your favorite fast food restaurant?",
    "What is your favorite sandwich?",
    "What is your favorite breakfast food?",
    "What is your favorite kind of bread?",
    "What is your favorite type of cheese?",
    "What is your favorite type of meat?",
    "What is your favorite type of fish?",
    "What is your favorite type of coffee?",
    "What is your favorite kind of tea?",
    "What is your favorite kind of juice?",
    "What is your favorite kind of soda?",
    "What is your favorite kind of pie?",
    "What is your favorite kind of cake?",
    "What is your favorite kind of cookie?",
    "What is your favorite kind of cereal?",
    "What is your favorite kind of pasta?",
    "What is your favorite kind of rice?",
    "What is your favorite kind of soup?",
    "What is your favorite kind of salad?",
    "What is your favorite kind of sandwich topping?",
    "What is your favorite kind of sauce?",
    "What is your favorite kind of spice?",
    "What is your favorite holiday tradition?",
    "What is your favorite way to relax?",
    "What is your favorite way to exercise?",
    "What is your favorite way to travel?",
    "What is your favorite way to celebrate a birthday?",
    "What is your favorite way to spend a weekend?",
    "What is your favorite way to spend a day off?",
    "What is your favorite way to stay healthy?",
    "What is your favorite way to learn new things?",
    "What is your favorite way to give back to the community?",
    "What is your favorite way to spend time with family?",
    "What is your favorite way to spend time with friends?",
    "What is your favorite way to enjoy nature?",
    "What is your favorite way to express creativity?",
    "What is your favorite way to handle stress?",
    "What is your favorite way to set goals?",
    "What is your favorite way to track progress?",
    "What is your favorite way to reward yourself?",
    "What is your favorite way to stay motivated?",
    "What is your favorite way to organize your day?",
    "What is your favorite way to prioritize tasks?",
    "What is your favorite way to manage time?",
    "What is your favorite way to communicate?",
    "What is your favorite way to solve problems?",
    "What is your favorite way to make decisions?",
    "What is your favorite way to overcome challenges?",
    "What is your favorite way to learn from mistakes?",
    "What is your favorite way to celebrate achievements?",
    "What is your favorite way to set boundaries?",
    "What is your favorite way to practice mindfulness?",
    "What is your favorite way to meditate?",
    "What is your favorite way to journal?",
    "What is your favorite way to read?",
    "What is your favorite way to watch movies?",
    "What is your favorite way to listen to music?",
    "What is your favorite way to engage in hobbies?",
    "What is your favorite way to stay informed?",
    "What is your favorite way to use technology?",
    "What is your favorite way to disconnect from technology?",
    "What is your favorite way to spend a rainy day?",
    "What is your favorite way to spend a sunny day?",
    "What is your favorite way to spend a snowy day?",
    "What is your favorite way to spend a windy day?",
    "What is your favorite way to spend a cloudy day?",
    "What is your favorite way to spend a hot day?",
    "What is your favorite way to spend a cold day?",
    "What is your favorite way to spend a long weekend?",
    "What is your favorite way to spend a short weekend?",
    "What is your favorite way to celebrate a holiday?",
    "What is your favorite way to decorate for a holiday?",
    "What is your favorite way to gift wrap presents?",
    "What is your favorite way to host a party?",
    "What is your favorite way to attend a party?",
    "What is your favorite way to prepare a meal?",
    "What is your favorite way to cook a meal?",
    "What is your favorite way to bake?",
    "What is your favorite way to grill?",
    "What is your favorite way to make a cake?",
    "What is your favorite way to make coffee?",
    "What is your favorite way to make tea?",
    "What is your favorite way to make a smoothie?",
    "What is your favorite way to make a cocktail?",
    "What is your favorite way to make a mocktail?",
    "What is your favorite way to make a sandwich?",
    "What is your favorite way to make a salad?",
    "What is your favorite way to make a soup?",
    "What is your favorite way to make a pizza?",
    "What is your favorite way to make pasta?",
    "What is your favorite way to make rice?",
    "What is your favorite way to make stir-fry?",
    "What is your favorite way to make a dessert?",
    "What is your favorite way to make a snack?",
    "What is your favorite way to make a breakfast dish?",
    "What is your favorite way to make a lunch dish?",
    "What is your favorite way to make a dinner dish?",
    "What is your favorite way to make a beverage?",
    "What is your favorite way to make a sauce?",
    "What is your favorite way to make a dip?",
    "What is your favorite way to make a dressing?",
    "What is your favorite way to make a marinade?",
    "What is your favorite way to make bread?",
    "What is your favorite way to make pastry?",
    "What is your favorite way to make a pie?",
    "What is your favorite way to make a cookie?",
    "What is your favorite way to make frosting?",
    "What is your favorite way to make filling?",
    "What is your favorite way to make topping?",
    "What is your favorite way to make a hot drink?",
    "What is your favorite way to make a cold drink?",
    "What is your favorite way to make a martini?",
    "What is your favorite way to make a margarita?",
    "What is your favorite way to make a daiquiri?",
    "What is your favorite way to make a mojito?",
    "What is your favorite way to make an old fashioned?",
    "What is your favorite way to make a cosmopolitan?",
    "What is your favorite way to make a mai tai?",
    "What is your favorite way to make a pina colada?",
    "What is your favorite way to make a screwdriver?",
    "What is your favorite way to make a bloody mary?",
    "What is your favorite way to make a moscow mule?",
    "What is your favorite way to make a gin and tonic?",
    "What is your favorite way to make a rum and coke?",
    "What is your favorite way to make a whiskey sour?",
    "What is your favorite way to make a negroni?",
    "What is your favorite way to make a sangria?",
    "What is your favorite way to make a spritz?",
    "What is your favorite way to make a bellini?",
    "What is your favorite way to make a mimosa?",
    "What is your favorite way to make a french 75?",
    "What is your favorite way to make a tom collins?",
    "What is your favorite way to make a sidecar?",
    "What is your favorite way to make a whiskey neat?",
    "What is your favorite way to make a mixed drink?",
    "What is your favorite way to make a highball?",
    "What is your favorite way to make a lowball?",
    "What is your favorite way to make a shot?",
    "What is your favorite way to make a layered drink?",
    "What is your favorite way to make a frozen cocktail?",
    "What is your favorite way to make a hot cocktail?",
    "What is your favorite way to make a sour cocktail?",
    "What is your favorite way to make a sweet cocktail?",
    "What is your favorite way to make a bitter cocktail?",
    "What is your favorite way to make a fruity cocktail?",
    "What is your favorite way to make a spicy cocktail?",
    "What is your favorite way to make a savory cocktail?",
    "What is your favorite way to make a smoky cocktail?",
    "What is your favorite way to make a herbaceous cocktail?",
    "What is your favorite way to make a floral cocktail?",
    "What is your favorite way to make a citrus cocktail?",
    "What is your favorite way to make a tropical cocktail?",
    "What is your favorite way to make a classic cocktail?",
    "What is your favorite way to make a signature cocktail?",
    "What is your favorite way to make a customized cocktail?",
    "What is your favorite way to make a party drink?",
    "What is your favorite way to make a festive drink?",
    "What is your favorite way to make a special occasion drink?",
    "What is your favorite way to make a holiday cocktail?",
    "What is your favorite way to make a themed cocktail?",
    "What is your favorite way to make a new cocktail?",
    "What is your favorite way to make an old-fashioned drink?",
    "What is your favorite memory from childhood?",
    "What is the most adventurous thing you've ever done?",
    "What is a skill you’d like to learn and why?",
    "What inspires you the most?",
    "What is your favorite quote or saying?",
    "If you could travel anywhere in the world, where would you go?",
    "What is your favorite way to spend a rainy day?",
    "What is your favorite way to spend a sunny day?",
    "What is your favorite childhood game?",
    "Who is your role model and why?",
    "What is your favorite family tradition?",
    "What is the best piece of advice you've ever received?",
    "What accomplishment are you most proud of?",
    "What is your favorite thing about your current job?",
    "What motivates you to work hard?",
    "How do you handle failure?",
    "What are your long-term career goals?",
    "How do you stay organized?",
    "What is your preferred work environment?",
    "How do you prioritize tasks?",
    "What is your greatest strength?",
    "What is your greatest weakness?",
    "How do you handle stress?",
    "Describe a challenging situation and how you overcame it.",
    "What makes you unique?",
    "Why do you want to join our organization?",
    "What can you bring to our team?",
    "Describe your ideal job.",
    "How do you handle conflict?",
    "What are your hobbies outside of work?",
    "How do you stay updated with industry trends?",
    "What is your preferred method of communication?",
    "Describe a time you demonstrated leadership.",
    "What is your favorite project you’ve worked on?",
    "How do you set goals for yourself?",
    "What do you value most in a workplace?",
    "How do you balance work and personal life?",
    "What is your favorite way to unwind after work?",
    "Describe a time you worked in a team.",
    "What is your favorite technology tool?",
    "How do you approach problem-solving?",
    "What is your favorite way to receive feedback?",
    "Describe your perfect workday.",
    "What are you passionate about?",
    "How do you handle tight deadlines?",
    "What is your favorite leadership style?",
    "Describe a time you went above and beyond.",
    "What is your favorite aspect of your profession?",
    "How do you continue to grow professionally?",
    "What is your favorite way to learn new skills?",
    "Describe a time you had to adapt to change.",
    "What is your favorite way to collaborate with others?",
    "How do you handle multiple responsibilities?",
    "What is your favorite motivational book?",
    "Describe a time you received constructive criticism.",
    "What is your favorite way to celebrate success?",
    "How do you ensure quality in your work?",
    "What is your favorite professional achievement?",
    "Describe a time you had to meet a difficult goal.",
    "What is your favorite aspect of our company?",
    "How do you contribute to a positive team environment?",
    "What is your favorite way to manage time effectively?",
    "Describe a time you had to learn something new quickly.",
    "What is your favorite way to stay motivated during challenges?",
    "How do you handle unexpected changes in a project?",
    "What is your favorite strategy for overcoming obstacles?",
    "Describe a time you improved a process.",
    "What is your favorite way to mentor others?",
    "How do you stay focused on your tasks?",
    "What is your favorite way to celebrate milestones?",
    "Describe a time you had to make a tough decision.",
    "What is your favorite way to give back to the community?",
    "How do you handle working under pressure?",
    "What is your favorite way to achieve work-life balance?",
    "Describe a time you successfully managed a project.",
    "What is your favorite way to build relationships at work?",
    "How do you stay inspired in your role?",
    "What is your favorite way to contribute to team goals?",
    "Describe a time you had to persuade someone.",
    "What is your favorite way to handle feedback?",
    "How do you prioritize your professional development?",
    "What is your favorite way to tackle a new challenge?",
    "Describe a time you demonstrated resilience.",
    "What is your favorite way to innovate in your work?",
    "How do you maintain a positive attitude?",
    "What is your favorite way to celebrate diversity at work?",
    "Describe a time you led a successful initiative.",
    "What is your favorite way to stay productive?",
    "How do you handle competing priorities?",
    "What is your favorite way to contribute to company culture?",
    "Describe a time you exceeded expectations.",
    "What is your favorite way to engage with clients/customers?",
    "How do you ensure continuous improvement in your work?",
    "What is your favorite way to develop new ideas?",
    "Describe a time you collaborated with a difficult colleague.",
    "What is your favorite way to achieve your goals?",
    "How do you stay adaptable in a changing environment?",
    "What is your favorite way to demonstrate integrity at work?",
    "Describe a time you took initiative on a project.",
    "What is your favorite way to celebrate team achievements?",
    "How do you handle constructive criticism?",
    "What is your favorite way to foster teamwork?",
    "Describe a time you successfully managed conflict.",
    "What is your favorite way to contribute to your team's success?",
]  

num_unique = len(unique_questions)
num_samples = 200 - num_unique

questions = unique_questions + [f"Sample question {i}" for i in range(1, num_samples + 1)]
# ###############################################################################################################################
# ###############################################################################################################################

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

# ###############################################################################################################################
# ###############################################################################################################################

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
# ###############################################################################################################################
# ###############################################################################################################################


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
# ###############################################################################################################################
# ###############################################################################################################################

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
# ###############################################################################################################################
# ###############################################################################################################################

@app.route('/login')
def login():
    return render_template('login.html')

# ###############################################################################################################################
# ###############################################################################################################################


@app.route('/authenticate', methods=['POST'])
def authenticate():
    user_id = request.form['user_id'].strip()

    if not user_id:
        return render_template('login.html', message="Username cannot be empty.", category="error")

    # Check if user exists
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return render_template('login.html', message="Username not found. Please enroll first.", category="error")

    # Check if user has any questions
    if not user.get('answers'):
        return render_template('login.html', message="No enrolled questions found. Please enroll first.", category="error")

    # Select one random question from the enrolled questions
    enrolled_questions = list(user['answers'].keys())  # Should be ['q1', 'q2', 'q3']
    selected_question = random.choice(enrolled_questions)

    # Store the selected question in session
    session['user_id'] = user_id
    session['selected_question'] = selected_question

    # Pass the question text to the template
    return render_template('verify.html', question=user['answers'][selected_question]['question_text'])
# ###############################################################################################################################
# ###############################################################################################################################


@app.route('/secure_dashboard')
def secure_dashboard():
    user_id = session.get('user_id')
    if not user_id:
        logger.info("No user_id in session. Redirecting to login.")
        return redirect(url_for('login'))
    logger.info(f"Rendering secure_dashboard for user_id: {user_id}")
    return render_template('secure_dashboard.html', user_id=user_id)

# ###############################################################################################################################

# ###############################################################################################################################

# @app.route('/verify_answer', methods=['POST'])
# def verify_answer():
#     try:
#         # Check if user_id and selected_question are in session
#         if 'user_id' not in session or 'selected_question' not in session:
#             logger.info("Session missing user_id or selected_question. Redirecting to login.")
#             return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401

#         user_id = session['user_id']
#         question_key = session['selected_question']  # Should be 'q1', 'q2', etc.
#         audio_file = request.files.get('audio')

#         # Check if audio file is provided
#         if not audio_file:
#             logger.error("No audio file provided.")
#             return jsonify({'status': 'error', 'message': 'No audio file provided.'}), 400

#         # Read audio data
#         audio_data = audio_file.read()

#         # Convert audio to WAV format
#         logger.info("Converting audio to WAV...")
#         wav_data = convert_audio_to_wav(audio_data)
#         if not wav_data:
#             logger.error("Audio conversion failed.")
#             return jsonify({'status': 'error', 'message': 'Audio conversion failed.'}), 500

#         # Transcribe audio
#         logger.info("Transcribing audio...")
#         transcription = transcribe_audio(wav_data)
#         if transcription.strip() == "":
#             logger.error("Transcription failed or was empty.")
#             return jsonify({'status': 'error', 'message': 'Transcription failed or was empty.'}), 500
#         print('transcript----------------------')
#         print(transcription.strip().lower())
#         print(type(transcription))
#         print('--------------------------------')
# # ###############################################################################################################################
#        # Compute embedding
#         logger.info("Computing embedding...")
#         embedding = compute_embedding(wav_data)
#         if embedding is None:
#             logger.error("Embedding computation failed.")
#             return jsonify({'status': 'error', 'message': 'Embedding computation failed.'}), 500

#         # Convert embedding to list for comparison
#         embedding_list = embedding.tolist()
# # ###############################################################################################################################

#         # Retrieve user data from the database
#         logger.info("Retrieving user data from the database...")
#         user = users_collection.find_one({"user_id": user_id})
#         if not user:
#             logger.error(f"User {user_id} not found.")
#             return jsonify({'status': 'error', 'message': 'User not found.'}), 404
# # ###############################################################################################################################

#         # Retrieve stored embeddings for the selected question
#         question_data = user.get('answers', {}).get(question_key, {})
#         stored_embeddings = [emb['embedding'] for emb in question_data.get('embeddings', [])]
#         expected_transcription = question_data.get('transcription', "").strip().lower()
        
        
#         print('right aswer---------------------')
#         print(expected_transcription)
#         print(repr(expected_transcription))
#         print(type(expected_transcription))
#         print('--------------------------------')
#         if not stored_embeddings:
#             logger.error(f"No stored embeddings found for question {question_key}.")
#             return jsonify({'status': 'error', 'message': 'No stored embeddings found for this question.'}), 404
# # ###############################################################################################################################
#        # Compute cosine similarities
#         logger.info("Computing cosine similarities...")
#         similarities = compute_cosine_similarity(embedding_list, stored_embeddings)

#         # Define similarity threshold
#         THRESHOLD = 0.10  # Adjust based on your requirements

#         # Find the maximum similarity
#         max_similarity = max(similarities)
#         logger.info(f"Max Similarity: {max_similarity}")

#         # Log the result
#         logger.info(f"Max Similarity for user {user_id} on question '{question_data.get('question_text', 'Unknown')}': {max_similarity}")
# # ###############################################################################################################################
#         transcription_pass = transcription == expected_transcription
#         logger.info(f"Transcription Pass: {transcription_pass}")
#         # Check if any similarity exceeds the threshold
#         if max_similarity >= THRESHOLD and transcription_pass:
#             # Successful authentication
#             logger.info("Authentication successful.")
#             return jsonify({'status': 'success', 'transcription': transcription, 'result': 'open'}), 200
#         else:
#             # Authentication failed
#             logger.info("Authentication failed.")
#             return jsonify({'status': 'error', 'message': 'Authentication failed.'}), 401

#     except Exception as e:
#         logger.exception("An unexpected error occurred during verification.")
#         return jsonify({'status': 'error', 'message': 'An unexpected error occurred.'}), 500
# ###############################################################################################################################

@app.route('/verify_answer', methods=['POST'])
def verify_answer():
    try:
        # Check if user_id and selected_question are in session
        if 'user_id' not in session or 'selected_question' not in session:
            logger.info("Session missing user_id or selected_question. Redirecting to login.")
            return jsonify({'status': 'error', 'message': 'Session expired. Please log in again.'}), 401

        user_id = session['user_id']
        question_key = session['selected_question']  # Should be 'q1', 'q2', etc.
        audio_file = request.files.get('audio')

        # Check if audio file is provided
        if not audio_file:
            logger.error("No audio file provided.")
            return jsonify({'status': 'error', 'message': 'No audio file provided.'}), 400

        # Read audio data
        audio_data = audio_file.read()

        # Convert audio to WAV format
        logger.info("Converting audio to WAV...")
        wav_data = convert_audio_to_wav(audio_data)
        if not wav_data:
            logger.error("Audio conversion failed.")
            return jsonify({'status': 'error', 'message': 'Audio conversion failed.'}), 500

        # Transcribe audio
        logger.info("Transcribing audio...")
        transcription = transcribe_audio(wav_data)
        if transcription.strip() == "":
            logger.error("Transcription failed or was empty.")
            return jsonify({'status': 'error', 'message': 'Transcription failed or was empty.'}), 500
        
        # Normalize transcription
        normalized_transcription = transcription.strip().lower()
        logger.info(f"Transcription: {repr(normalized_transcription)}")

        # Compute embedding
        logger.info("Computing embedding...")
        embedding = compute_embedding(wav_data)
        if embedding is None:
            logger.error("Embedding computation failed.")
            return jsonify({'status': 'error', 'message': 'Embedding computation failed.'}), 500

        # Convert embedding to list for comparison
        embedding_list = embedding.tolist()

        # Retrieve user data from the database
        logger.info("Retrieving user data from the database...")
        user = users_collection.find_one({"user_id": user_id})
        if not user:
            logger.error(f"User {user_id} not found.")
            return jsonify({'status': 'error', 'message': 'User not found.'}), 404

        # Retrieve stored embeddings for the selected question
        question_data = user.get('answers', {}).get(question_key, {})
        stored_embeddings = [emb['embedding'] for emb in question_data.get('embeddings', [])]
        expected_transcription = question_data.get('transcription', "").strip().lower()

        logger.info(f"Expected Transcription: {repr(expected_transcription)}")

        if not stored_embeddings:
            logger.error(f"No stored embeddings found for question {question_key}.")
            return jsonify({'status': 'error', 'message': 'No stored embeddings found for this question.'}), 404

        # Compute cosine similarities
        logger.info("Computing cosine similarities...")
        similarities = compute_cosine_similarity(embedding_list, stored_embeddings)

        # Define similarity threshold
        THRESHOLD = 0.10  # Adjust based on your requirements

        # Find the maximum similarity
        max_similarity = max(similarities) if similarities else 0
        logger.info(f"Max Similarity: {max_similarity}")

        # Check transcription match
        transcription_pass = normalized_transcription == expected_transcription
        logger.info(f"Transcription Pass: {transcription_pass}")

        # Check if any similarity exceeds the threshold
        similarity_pass = max_similarity >= THRESHOLD
        logger.info(f"Similarity Pass: {similarity_pass}")

        # if similarity_pass and transcription_pass:
        #     # Successful authentication
        #     logger.info("Authentication successful.")
        #     return jsonify({'status': 'success', 'transcription': transcription, 'result': 'open'}), 200
        if similarity_pass:
            # Successful authentication
            logger.info("Authentication successful.")
            return jsonify({'status': 'success', 'transcription': transcription, 'result': 'open'}), 200
        else:
            # Authentication failed
            failure_reasons = []
            if not similarity_pass:
                failure_reasons.append("Voice does not match.")
            if not transcription_pass:
                failure_reasons.append("Transcription does not match expected answer.")
            logger.info("Authentication failed.")
            return jsonify({'status': 'error', 'message': ' '.join(failure_reasons)}), 401

    except Exception as e:
        logger.exception("An unexpected error occurred during verification.")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred.'}), 500

# ###############################################################################################################################

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
# ###############################################################################################################################
# ###############################################################################################################################


@app.route('/')
def home():
    return render_template('index.html')
# ###############################################################################################################################
# ###############################################################################################################################

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
# ###############################################################################################################################
# ###############################################################################################################################

@app.route('/get_questions', methods=['GET'])
def get_questions():
    selected_questions = random.sample(questions, 3)
    return jsonify({"questions": selected_questions})
# ###############################################################################################################################
# ###############################################################################################################################


@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    user_id = request.form['user_id']
    question_text = request.form['question']
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
    if not user:
        return jsonify({"status": "error", "message": "User not found."}), 404

    # Assigning to q1, q2, q3 based on existing entries
    existing_questions = user.get('answers', {})
    if len(existing_questions) < 3:
        q_key = f"q{len(existing_questions) + 1}"
        users_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    f"answers.{q_key}.question_text": question_text,
                    f"answers.{q_key}.answer": transcription.lower(),
                    f"answers.{q_key}.embeddings": [embedding_data]
                }
            },
            upsert=True
        )
    else:
        # Update existing question's embeddings
        for q_key, q_data in existing_questions.items():
            if q_data.get('question_text') == question_text:
                users_collection.update_one(
                    {"user_id": user_id},
                    {"$push": {f"answers.{q_key}.embeddings": embedding_data}},
                    upsert=True
                )
                break
        else:
            return jsonify({"status": "error", "message": "Question not found."}), 400

    # Check if the user has answered all three questions
    user = users_collection.find_one({"user_id": user_id})
    if len(user.get('answers', {})) >= 3:
        return jsonify({"status": "complete", "message": "Enrollment complete."})

    return jsonify({"status": "success", "transcription": transcription.lower(), "recording_id": recording_id})

# ###############################################################################################################################
# ###############################################################################################################################

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
# ###############################################################################################################################
# ###############################################################################################################################

@app.route('/complete/<user_id>')
def complete(user_id):
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return "User not found", 404
    return render_template('complete.html', user=user)
# ###############################################################################################################################
# ###############################################################################################################################

@app.route('/cancel_enrollment', methods=['POST'])
def cancel_enrollment():
    user_id = request.form['user_id']
    users_collection.delete_one({"user_id": user_id})
    return jsonify({"status": "success", "message": "Enrollment canceled and data deleted."})

# ###############################################################################################################################
# ###############################################################################################################################

@app.route('/encrypt')
def encrypt():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('encrypt.html')

# ###############################################################################################################################
# ###############################################################################################################################

@app.route('/process_encryption', methods=['POST'])
def process_encryption():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    ans1 = request.form.get('ans1', '').strip()
    ans2 = request.form.get('ans2', '').strip()
    ans3 = request.form.get('ans3', '').strip()
    uploaded_file = request.files.get('file', None)
    
    if not all([ans1, ans2, ans3, uploaded_file]):
        return render_template('encrypt.html', message="All fields are required.", category="error")
    
    # Retrieve user data
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return render_template('encrypt.html', message="User not found.", category="error")
    
    stored_answers = user.get('answers', {})
    # Assuming answers are stored under 'q1', 'q2', 'q3'
    stored_ans1 = stored_answers.get('q1', {}).get('answer', '').strip().lower()
    stored_ans2 = stored_answers.get('q2', {}).get('answer', '').strip().lower()
    stored_ans3 = stored_answers.get('q3', {}).get('answer', '').strip().lower()
    
    # Verify answers
    if (ans1.lower() != stored_ans1) or (ans2.lower() != stored_ans2) or (ans3.lower() != stored_ans3):
        return render_template('encrypt.html', message="Incorrect answers to security questions.", category="error")
    
    # Save uploaded file to a temporary location
    filename = secure_filename(uploaded_file.filename)
    temp_dir = tempfile.gettempdir()
    input_filepath = os.path.join(temp_dir, filename)
    uploaded_file.save(input_filepath)
    
    # Determine file type and select appropriate encryption binary
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext == '.txt':
        cpp_encrypt_executable = './C++./a.out'  # Ensure the path is correct
        args = [cpp_encrypt_executable, input_filepath, ans1, ans2, ans3]
    elif file_ext == '.wav':
        cpp_encrypt_executable = './encrypt_audio'
        args = [cpp_encrypt_executable, input_filepath, ans1, ans2, ans3]
    elif file_ext in ['.png', '.jpg', '.jpeg']:
        cpp_encrypt_executable = './encrypt_image'
        args = [cpp_encrypt_executable, input_filepath, ans1, ans2, ans3]
    else:
        return render_template('encrypt.html', message="Unsupported file type.", category="error")
    
    try:
        start_time = time.time()
        subprocess.run(args, check=True)
        end_time = time.time()
        
        # Determine output filename based on encryption binary
        if file_ext == '.txt':
            output_filename = 'encrypted_' + filename
        elif file_ext == '.wav':
            output_filename = 'encrypted_audio_' + filename
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            output_filename = 'encrypted_image_' + filename
        else:
            output_filename = 'encrypted_' + filename  # Fallback
        
        output_filepath = os.path.join(temp_dir, output_filename)
        
        if not os.path.exists(output_filepath):
            return render_template('encrypt.html', message="Encryption failed. Output file not found.", category="error")
        
        # Read encrypted file
        with open(output_filepath, 'rb') as f:
            encrypted_data = f.read()
        
        # Clean up temporary files
        os.remove(input_filepath)
        os.remove(output_filepath)
        
        # Provide encrypted file for download
        return send_file(
            io.BytesIO(encrypted_data),
            as_attachment=True,
            download_name=output_filename
        )
        
    except subprocess.CalledProcessError as e:
        print(f"Encryption Error: {e}")
        return render_template('encrypt.html', message="An error occurred during encryption.", category="error")
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return render_template('encrypt.html', message="An unexpected error occurred.", category="error")
# ###############################################################################################################################
# ###############################################################################################################################

@app.route('/decrypt')
def decrypt():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('decrypt.html')
# ###############################################################################################################################
# ###############################################################################################################################

@app.route('/process_decryption', methods=['POST'])
def process_decryption():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    ans1 = request.form.get('ans1', '').strip()
    ans2 = request.form.get('ans2', '').strip()
    ans3 = request.form.get('ans3', '').strip()
    uploaded_file = request.files.get('file', None)
    
    if not all([ans1, ans2, ans3, uploaded_file]):
        return render_template('decrypt.html', message="All fields are required.", category="error")
    
    # Retrieve user data
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return render_template('decrypt.html', message="User not found.", category="error")
    
    stored_answers = user.get('answers', {})
    # Assuming answers are stored under 'q1', 'q2', 'q3'
    stored_ans1 = stored_answers.get('q1', {}).get('answer', '').strip().lower()
    stored_ans2 = stored_answers.get('q2', {}).get('answer', '').strip().lower()
    stored_ans3 = stored_answers.get('q3', {}).get('answer', '').strip().lower()
    
    # Verify answers
    if (ans1.lower() != stored_ans1) or (ans2.lower() != stored_ans2) or (ans3.lower() != stored_ans3):
        return render_template('decrypt.html', message="Incorrect answers to security questions.", category="error")
    
    # Save uploaded file to a temporary location
    filename = secure_filename(uploaded_file.filename)
    temp_dir = tempfile.gettempdir()
    input_filepath = os.path.join(temp_dir, filename)
    uploaded_file.save(input_filepath)
    
    # Determine file type and select appropriate decryption binary
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext == '.txt' and filename.startswith('encrypted_'):
        cpp_decrypt_executable = './dec_txt'
        args = [cpp_decrypt_executable, input_filepath, ans1, ans2, ans3]
    elif file_ext == '.wav' and filename.startswith('encrypted_audio_'):
        cpp_decrypt_executable = './decrypt_audio'
        args = [cpp_decrypt_executable, input_filepath, ans1, ans2, ans3]
    elif file_ext in ['.png', '.jpg', '.jpeg'] and filename.startswith('encrypted_image_'):
        cpp_decrypt_executable = './decrypt_image'
        args = [cpp_decrypt_executable, input_filepath, ans1, ans2, ans3]
    else:
        return render_template('decrypt.html', message="Unsupported or improperly named file type.", category="error")
    
    try:
        start_time = time.time()
        subprocess.run(args, check=True)
        end_time = time.time()
        
        # Determine output filename based on decryption binary
        if file_ext == '.txt' and filename.startswith('encrypted_'):
            output_filename = 'decrypted_' + filename[len('encrypted_'):]
        elif file_ext == '.wav' and filename.startswith('encrypted_audio_'):
            output_filename = 'decrypted_audio_' + filename[len('encrypted_audio_'):]
        elif file_ext in ['.png', '.jpg', '.jpeg'] and filename.startswith('encrypted_image_'):
            output_filename = 'decrypted_image_' + filename[len('encrypted_image_'):]
        else:
            output_filename = 'decrypted_' + filename  # Fallback
        
        output_filepath = os.path.join(temp_dir, output_filename)
        
        if not os.path.exists(output_filepath):
            return render_template('decrypt.html', message="Decryption failed. Output file not found.", category="error")
        
        # Read decrypted file
        with open(output_filepath, 'rb') as f:
            decrypted_data = f.read()
        
        # Clean up temporary files
        os.remove(input_filepath)
        os.remove(output_filepath)
        
        # Provide decrypted file for download
        return send_file(
            io.BytesIO(decrypted_data),
            as_attachment=True,
            download_name=output_filename
        )
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Decryption Error: {e}")
        return render_template('decrypt.html', message="An error occurred during decryption.", category="error")
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return render_template('decrypt.html', message="An unexpected error occurred.", category="error")
# ###############################################################################################################################
# ###############################################################################################################################
# ###############################################################################################################################
# ###############################################################################################################################

if __name__ == '__main__':
    app.run(debug=True)
