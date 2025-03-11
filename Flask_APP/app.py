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
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration


# import spacy

import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


app = Flask(__name__)
app.secret_key = os.urandom(24)


# Allowed extensions and their corresponding encryption executables and output patterns
# Allowed extensions and their corresponding encryption executables and output patterns

SUPPORTED_OUTPUT_TYPES = {
    'txt': 'txt',
    'jpg': 'jpg',
    'jpeg': 'jpeg',
    'png': 'png',
    'mp3': 'mp3',
    'wav': 'wav'
}
ALLOWED_EXTENSIONS = {
    'txt': {
        'executable': os.path.abspath('./C++/enc_txt'),
        'output_pattern': 'encrypted_{filename}'  # Changed from 'encrypted_{filename}.txt' to 'encrypted_{filename}'
    },
    'jpg': {
        'executable': os.path.abspath('./C++/enc_img'),
        'output_pattern': 'encrypted_{filename}.txt'
    },
    'jpeg': {
        'executable': os.path.abspath('./C++/enc_img'),
        'output_pattern': 'encrypted_{filename}.txt'
    },
    'png': {
        'executable': os.path.abspath('./C++/enc_img'),
        'output_pattern': 'encrypted_{filename}.txt'
    },
    'mp3': {
        'executable': os.path.abspath('./C++/enc_audio'),
        'output_pattern': 'encrypted_audio_data.txt'  # As per current behavior
    },
    'wav': {
        'executable': os.path.abspath('./C++/enc_audio'),
        'output_pattern': 'encrypted_audio_data.txt'  # As per current behavior
    }
}

# Path to the decryption executables
DECRYPTION_EXECUTABLES = {
    'txt': os.path.abspath('./C++/dec_txt'),
    'jpg': os.path.abspath('./C++/dec_img'),
    'jpeg': os.path.abspath('./C++/dec_img'),
    'png': os.path.abspath('./C++/dec_img'),
    'mp3': os.path.abspath('./C++/dec_audio'),
    'wav': os.path.abspath('./C++/dec_audio')
}

DECRYPTION_EXTENSIONS = {
    'txt': {
        'executable': os.path.abspath('./C++/dec_txt'),
        'output_pattern': 'decrypted_{filename}'
    },
    'jpg': {
        'executable': os.path.abspath('./C++/dec_img'),
        'output_pattern': 'decrypted_{filename}.jpg'
    },
    'jpeg': {
        'executable': os.path.abspath('./C++/dec_img'),
        'output_pattern': 'decrypted_{filename}.jpeg'
    },
    'png': {
        'executable': os.path.abspath('./C++/dec_img'),
        'output_pattern': 'decrypted_{filename}.png'
    },
    'mp3': {
        'executable': os.path.abspath('./C++/dec_audio'),
        'output_pattern': 'decrypted_audio_data.mp3'
    },
    'wav': {
        'executable': os.path.abspath('./C++/dec_audio'),
        'output_pattern': 'decrypted_audio_data.wav'
    }
}


print("mongo-Db..--------------")
print('------------------------')
print('------------------------')
app.config["MONGO_URI"] = "mongodb://localhost:27017/voice_authentication_system"
mongo = PyMongo(app)
users_collection = mongo.db.users
print('------------------------')
print('------------------------')


print("transformer...----------")
print('------------------------')
print('------------------------')
model_name = "google/flan-t5-small"  # Use "flan-t5-base" or "flan-t5-large" for better results
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

print('------------------------')
print('------------------------')

# print("spacy loaded..----------")
# print('------------------------')
# print('------------------------')
# nlp = spacy.load("en_core_web_sm")
# print('------------------------')
# print('------------------------')

print("Loading Whisper model...")
print('------------------------')
print('------------------------')
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")
print('------------------------')
print('------------------------')


print("Loading SpeakerRecognition model...")
print('------------------------')
print('------------------------')
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")
print("SpeakerRecognition model loaded.")
print('------------------------')
print('------------------------')


print('------------------------')
print('uniquestion loaded')
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
ALLOWED_ENCRYPTED_EXTENSIONS = {'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_ENCRYPTED_EXTENSIONS

def allowed_decryption_file_type(file_type):
    return file_type.lower() in DECRYPTION_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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


def cleanup_temp_dir(temp_dir):
    """Delete the temporary directory and its contents."""
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Temporary directory {temp_dir} deleted successfully.")
    except Exception as e:
        logger.error(f"Error deleting temporary directory {temp_dir}: {e}")



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

def update_embedding_incrementally(old_embedding, new_embedding, alpha=0.9):
    """Update embedding by combining old and new embeddings."""
    updated_embedding = alpha * torch.tensor(old_embedding) + (1 - alpha) * torch.tensor(new_embedding)
    # Normalize the updated embedding
    updated_embedding = updated_embedding / torch.norm(updated_embedding)
    print(f"New Embeddings: {updated_embedding}")

    return updated_embedding.tolist()  # Convert back to list for JSON serialization

# ###############################################################################################################################
# ###############################################################################################################################

# def extract_answer_spacy(transcription, question_key, user_data):
#     """
#     Extract the most relevant single-word answer from the transcription using spaCy.
#     """
#     transcription = transcription.lower()
#     doc = nlp(transcription)

#     # Map question_key to actual question text
#     question_text = user_data['answers'][question_key]['question_text'].lower()

#     # Define possible answers based on question context
#     possible_answers = []

#     # Predefined keywords and patterns
#     colors = {"red", "blue", "green", "yellow", "purple", "orange", "black", "white", "gray", "pink"}
    
#     if "favorite color" in question_text:
#         # Extract adjectives or predefined color names
#         for token in doc:
#             if token.pos_ == "ADJ" or token.text in colors:
#                 possible_answers.append(token.text)

#     elif "pet's name" in question_text:
#         # Extract proper nouns or named entities labeled as PERSON
#         for ent in doc.ents:
#             if ent.label_ == "PERSON":
#                 possible_answers.append(ent.text)
#         for token in doc:
#             if token.pos_ in ["PROPN", "NOUN"]:
#                 possible_answers.append(token.text)

#     else:
#         # General fallback: prioritize nouns, proper nouns, and adjectives
#         for token in doc:
#             if token.pos_ in ["NOUN", "PROPN", "ADJ"]:
#                 possible_answers.append(token.text)

#     # Return the first valid answer or fallback to the last significant word
#     return possible_answers[0] if possible_answers else (doc[-1].text if doc else "")

# ###############################################################################################################################
# ###############################################################################################################################


# def extract_answer_with_t5(transcription, question):
#     """
#     Use FLAN-T5 to extract the most relevant single-word or short-phrase answer.
#     """
#     prompt = f"""
#     Extract the single most relevant word or concise short phrase that directly answers the question.

#     Question: {question}
#     Transcription: {transcription}
#     Answer:
#     """
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

#     try:
#         outputs = model.generate(
#             inputs.input_ids,
#             max_length=10,  # Limit the output to a concise response
#             num_beams=3,    # Use beam search for better results
#             early_stopping=True
#         )
#         raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#         # Post-process to return a single word or short phrase joined by underscores
#         processed_answer = "_".join(raw_answer.split())
#         return processed_answer
#     except Exception as e:
#         logger.error(f"Error with FLAN-T5: {e}")
#         return None




def verify_answer_with_model(transcribed, expected):
    """
    Use the transformer model to decide if the transcribed answer matches the expected answer.
    Returns True if they mean the same thing, otherwise False.
    """
    prompt = (
        "Compare the following two sentences and decide if they convey the same meaning.\n"
        f"Sentence 1: '{transcribed}'\n"
        f"Sentence 2: '{expected}'\n"
        "Answer with 'True' or 'False'."
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    outputs = model.generate(
        inputs.input_ids,
        max_length=10,
        num_beams=3,
        early_stopping=True
    )
    # Decode and standardize the output
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    # You might encounter outputs like "true", "yes", etc.
    return "true" in raw_output



# ###############################################################################################################################
# ###############################################################################################################################


@app.route('/login')
def login():
    return render_template('login.html')
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
        wav_data = convert_audio_to_wav(audio_data)  # Ensure wav_data is defined
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
        logger.info(f"Total Number of embeddings compared: {len(stored_embeddings)}")
        logger.info(f"Old Embeddings: {stored_embeddings}")
        logger.info(f"Expected Transcription: {repr(expected_transcription)}")

        if not stored_embeddings:
            logger.error(f"No stored embeddings found for question {question_key}.")
            return jsonify({'status': 'error', 'message': 'No stored embeddings found for this question.'}), 404

        # Compute cosine similarities
        logger.info("Computing cosine similarities...")
        similarities = compute_cosine_similarity(embedding_list, stored_embeddings)

        # Define similarity threshold
        THRESHOLD = 0.60  # Adjust based on your requirements

        # Find the maximum similarity
        max_similarity = max(similarities) if similarities else 0
        logger.info(f"Max Similarity: {max_similarity}")

        # Use transformer model for verification of the transcription
        transcription_pass = verify_answer_with_model(normalized_transcription, expected_transcription)
        logger.info(f"Transcription verification (model) result: {transcription_pass}")

        # Check if any similarity exceeds the threshold
        similarity_pass = max_similarity >= THRESHOLD
        logger.info(f"Similarity Pass: {similarity_pass}")

        if transcription_pass and similarity_pass:
            # Successful authentication
            logger.info("Authentication successful.")

            # --- Embedding Update Logic Starts Here ---
            updated_embeddings = []
            for old_emb in stored_embeddings:
                updated_emb = update_embedding_incrementally(old_emb, embedding_list)
                updated_embeddings.append(updated_emb)

            # Optionally, limit the number of stored embeddings to prevent unbounded growth
            MAX_EMBEDDINGS = 10  # Example limit
            if len(updated_embeddings) > MAX_EMBEDDINGS:
                updated_embeddings = updated_embeddings[:MAX_EMBEDDINGS]

            # Update the embeddings in the database
            update_result = users_collection.update_one(
                {"user_id": user_id, f"answers.{question_key}": {"$exists": True}},
                {"$set": {f"answers.{question_key}.embeddings": [{"embedding": emb} for emb in updated_embeddings]}}
            )

            if update_result.modified_count == 1:
                logger.info("Embeddings successfully updated in the database.")
            else:
                logger.warning("Failed to update embeddings in the database.")
            # --- Embedding Update Logic Ends Here ---

            return jsonify({'status': 'success', 'transcription': transcription, 'result': 'open'}), 200
        else:
            # Authentication failed
            failure_reasons = []
            if not similarity_pass:
                failure_reasons.append("Voice does not match.")
            if not transcription_pass:
                failure_reasons.append("Answer verification failed.")
            logger.info("Authentication failed.")
            return jsonify({'status': 'error', 'message': ' '.join(failure_reasons)}), 401

    except Exception as e:
        logger.exception("An unexpected error occurred during verification.")
        return jsonify({'status': 'error', 'message': 'An unexpected error occurred.'}), 500
# ###############################################################################################################################
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
    user_id = request.args.get('user_id')
    reload = request.args.get('reload', 'false').lower() == 'true'  # New parameter

    if not user_id:
        return jsonify({'status': 'error', 'message': 'User ID is required.'}), 400

    # Reference the global `questions` variable
    global questions

    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return jsonify({'status': 'error', 'message': 'User not found.'}), 404

    if reload:
        # Clear existing answers to allow assignment of new questions
        users_collection.update_one({'user_id': user_id}, {'$unset': {'answers': ""}})
        answers = {}
    else:
        answers = user.get('answers', {})

    # Assign new questions if reload is requested or less than 3 questions are assigned
    if reload or len(answers) < 3:
        if not reload:
            # Exclude already assigned questions to ensure uniqueness
            assigned_questions = set(q['question_text'] for q in answers.values())
            available_questions = list(set(questions) - assigned_questions)
        else:
            # All questions are available since answers are cleared
            available_questions = questions.copy()

        if len(available_questions) < 3:
            return jsonify({'status': 'error', 'message': 'Not enough unique questions available.'}), 500

        selected_questions = random.sample(available_questions, 3)
        update = {}
        for i, q in enumerate(selected_questions, 1):
            update[f'answers.q{i}.question_text'] = q
            update[f'answers.q{i}.embeddings'] = []
            update[f'answers.q{i}.transcription'] = ""
        users_collection.update_one({'user_id': user_id}, {'$set': update})
        questions_list = selected_questions
    else:
        # Retrieve already assigned questions
        questions_list = [answers[f'q{i}']['question_text'] for i in range(1, 4)]

    return jsonify({'questions': questions_list})
# ###############################################################################################################################
# ###############################################################################################################################

# @app.route('/submit_answer', methods=['POST'])
# def submit_answer():
#     user_id = request.form.get('user_id', '').strip()
#     question_text = request.form.get('question', '').strip()
#     audio_file = request.files.get('audio')

#     if not all([user_id, question_text, audio_file]):
#         return jsonify({"status": "error", "message": "Missing parameters."}), 400

#     try:
#         # Read user data
#         user = users_collection.find_one({"user_id": user_id})
#         if not user:
#             return jsonify({"status": "error", "message": "User not found."}), 404

#         # Match the question text to the stored questions
#         answers = user.get('answers', {})
#         question_key = None
#         for key, value in answers.items():
#             if value.get('question_text', '').strip().lower() == question_text.strip().lower():
#                 question_key = key
#                 break

#         if not question_key:
#             return jsonify({'status': 'error', 'message': 'Question not found.'}), 404

#         # Process audio data (convert, transcribe, and compute embedding)
#         audio_data = audio_file.read()
#         wav_data = convert_audio_to_wav(audio_data)
#         if not wav_data:
#             return jsonify({'status': 'error', 'message': 'Audio conversion failed.'}), 500
        
#         transcription = transcribe_audio(wav_data)
#         if transcription.strip() == "":
#             return jsonify({'status': 'error', 'message': 'Transcription failed or was empty.'}), 500

#         embedding = compute_embedding(wav_data)
#         if embedding is None:
#             return jsonify({'status': 'error', 'message': 'Embedding computation failed.'}), 500

#         # Update the user's answer data
#         recording_id = str(uuid.uuid4())
#         embedding_data = {"recording_id": recording_id, "embedding": embedding.tolist()}
#         users_collection.update_one(
#             {"user_id": user_id},
#             {
#                 "$push": {f"answers.{question_key}.embeddings": embedding_data},
#                 "$set": {f"answers.{question_key}.transcription": transcription.strip().lower()},
#             }
#         )

#         return jsonify({"status": "success", "transcription": transcription, "recording_id": recording_id}), 200

#     except Exception as e:
#         logger.exception("Unexpected error in submit_answer.")
#         return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    user_id = request.form.get('user_id', '').strip()
    question_text = request.form.get('question', '').strip()
    audio_file = request.files.get('audio')

    if not all([user_id, question_text, audio_file]):
        return jsonify({"status": "error", "message": "Missing parameters."}), 400

    try:
        # Read user data
        user = users_collection.find_one({"user_id": user_id})
        if not user:
            return jsonify({"status": "error", "message": "User not found."}), 404

        # Match the question text to the stored questions
        answers = user.get('answers', {})
        question_key = None
        for key, value in answers.items():
            if value.get('question_text', '').strip().lower() == question_text.strip().lower():
                question_key = key
                break

        if not question_key:
            return jsonify({'status': 'error', 'message': 'Question not found.'}), 404

        # Process audio data (convert, transcribe, and compute embedding)
        audio_data = audio_file.read()
        wav_data = convert_audio_to_wav(audio_data)
        if not wav_data:
            return jsonify({'status': 'error', 'message': 'Audio conversion failed.'}), 500
        
        transcription = transcribe_audio(wav_data)
        if transcription.strip() == "":
            return jsonify({'status': 'error', 'message': 'Transcription failed or was empty.'}), 500

        logger.info(f"Transcription: {transcription}")

        embedding = compute_embedding(wav_data)
        if embedding is None:
            return jsonify({'status': 'error', 'message': 'Embedding computation failed.'}), 500

        # Use FLAN-T5 to extract the single-word or short-phrase answer
        processed_answer = extract_answer_with_t5(transcription, question_text)
        if not processed_answer:
            return jsonify({'status': 'error', 'message': 'Answer extraction failed.'}), 500

        logger.info(f"Processed answer: {processed_answer}")

        # Update the user's answer data
        recording_id = str(uuid.uuid4())
        embedding_data = {
            "recording_id": recording_id,
            "embedding": embedding.tolist(),
            "processed_answer": processed_answer  # Store the processed answer
        }
        users_collection.update_one(
            {"user_id": user_id},
            {
                "$push": {f"answers.{question_key}.embeddings": embedding_data},
                "$set": {f"answers.{question_key}.transcription": processed_answer},
            }
        )

        return jsonify({"status": "success", "transcription": transcription, "processed_answer": processed_answer, "recording_id": recording_id}), 200

    except Exception as e:
        logger.exception("Unexpected error in submit_answer.")
        return jsonify({"status": "error", "message": "An unexpected error occurred."}), 500

# ###############################################################################################################################
# ###############################################################################################################################
# ###############################################################################################################################
# ###############################################################################################################################

@app.route('/delete_recording', methods=['POST'])
def delete_recording():
    user_id = request.form.get('user_id')
    question_text = request.form.get('question')
    recording_id = request.form.get('recording_id')

    logger.info(f"Received delete_recording request for user_id: {user_id}, question: {question_text}, recording_id: {recording_id}")

    if not all([user_id, question_text, recording_id]):
        logger.warning("Missing parameters in delete_recording request.")
        return jsonify({"status": "error", "message": "Missing parameters."}), 400

    # Find the question key based on question_text
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        logger.warning(f"User {user_id} not found.")
        return jsonify({"status": "error", "message": "User not found."}), 404

    answers = user.get('answers', {})
    question_key = None
    for key, value in answers.items():
        if value.get('question_text') == question_text:
            question_key = key
            break

    if not question_key:
        logger.warning(f"Question '{question_text}' not found for user {user_id}.")
        return jsonify({"status": "error", "message": "Question not found."}), 404

    # Remove the embedding with the given recording_id
    result = users_collection.update_one(
        {"user_id": user_id},
        {"$pull": {f"answers.{question_key}.embeddings": {"recording_id": recording_id}}}
    )

    if result.modified_count > 0:
        logger.info(f"Recording {recording_id} deleted for user {user_id}, question {question_key}.")

        # Check if any embeddings remain
        user = users_collection.find_one({"user_id": user_id})
        embeddings = user.get('answers', {}).get(question_key, {}).get('embeddings', [])

        if not embeddings:
            # Reset transcription since no embeddings remain
            users_collection.update_one(
                {"user_id": user_id},
                {"$set": {f"answers.{question_key}.transcription": ""}}
            )
            logger.info(f"Transcription reset for question {question_key} as no embeddings remain.")

        return jsonify({"status": "success", "message": "Recording deleted."})
    else:
        logger.warning(f"Recording {recording_id} not found for user {user_id}, question {question_key}.")
        return jsonify({"status": "error", "message": "Recording not found."}), 404
# ###############################################################################################################################
# ###############################################################################################################################

@app.route('/finalize_enrollment', methods=['POST'])
def finalize_enrollment():
    data = request.get_json()
    user_id = data.get('user_id')

    if not user_id:
        return jsonify({"status": "error", "message": "User ID is required."}), 400

    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return jsonify({"status": "error", "message": "User not found."}), 404

    # Optionally, set a flag to indicate enrollment completion
    users_collection.update_one(
        {"user_id": user_id},
        {"$set": {"enrollment_complete": True}}
    )

    return jsonify({"status": "success", "message": "Enrollment finalized."})
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

@app.route('/authenticate', methods=['POST'])
def authenticate():
    user_id = request.form['user_id'].strip()

    if not user_id:
        return render_template('login.html', message="Username cannot be empty.", category="error")

    # Check if user exists in the database
    user = users_collection.find_one({"user_id": user_id})  # Use consistent key: `user_id`
    if not user:
        return render_template('login.html', message="Username not found. Please enroll first.", category="error")

    # Check if the user has enrolled questions and answers
    if not user.get('answers') or len(user.get('answers')) < 3:
        return render_template('login.html', message="Insufficient enrolled questions. Please enroll first.", category="error")

    # Get all enrolled questions
    enrolled_questions = list(user['answers'].keys())  # Typically, ['q1', 'q2', 'q3']
    if not enrolled_questions:
        return render_template('login.html', message="No enrolled questions found. Please enroll first.", category="error")

    # Randomly select one question
    selected_question = random.choice(enrolled_questions)

    # Store the selected question and user_id in the session
    session['user_id'] = user_id
    session['selected_question'] = selected_question

    # Retrieve the question text from the user's answers
    question_text = user['answers'][selected_question].get('question_text', "Unknown question")

    # Render the verify page with the selected question
    return render_template('verify.html', question=question_text)

# ###############################################################################################################################
# ###############################################################################################################################

# @app.route('/encrypt')
# def encrypt():
#     # Check if the user is logged in
#     if 'user_id' not in session:
#         logger.info("No user_id in session. Redirecting to login.")
#         return redirect(url_for('login'))

#     # Fetch user details using `user_id` from the session
#     user_id = session['user_id']
#     user = users_collection.find_one({"user_id": user_id})

#     # Check if the user exists and has completed enrollment
#     if not user or not user.get('answers') or len(user['answers']) < 3:
#         logger.info(f"User {user_id} not properly enrolled. Redirecting to secure dashboard.")
#         return render_template(
#             'secure_dashboard.html',
#             message="Please complete enrollment first.",
#             category="error",
#             user_id=user_id
#         )

#     # Log the user's ID and proceed to encryption page
#     logger.info(f"Rendering encrypt.html for user_id: {user_id}")

#     # Render the encrypt page without any unrelated variables
#     return render_template('encrypt.html', user_id=user_id)

# # ###############################################################################################################################
# # ###############################################################################################################################

# @app.route('/process_encryption', methods=['POST'])
# def process_encryption():
#     if 'user_id' not in session:
#         return redirect(url_for('login'))

#     # Fetch `user_id` from session
#     user_id = session['user_id']
#     uploaded_file = request.files.get('file', None)

#     if not uploaded_file:
#         return render_template(
#             'encrypt.html',
#             message="File is required.",
#             category="error",
#             user_id=user_id
#         )

#     # Fetch user data from MongoDB using `user_id`
#     user = users_collection.find_one({"user_id": user_id})
#     if not user:
#         return render_template(
#             'encrypt.html',
#             message="User not found. Please enroll first.",
#             category="error",
#             user_id=user_id
#         )

#     answers = user.get('answers', {})
#     if len(answers) < 3:
#         return render_template(
#             'encrypt.html',
#             message="Insufficient transcriptions for encryption. Please ensure at least 3 questions are enrolled.",
#             category="error",
#             user_id=user_id
#         )

#     # Extract transcriptions for the first 3 enrolled questions

#     # Path to the C++ executable
#     cpp_encrypt_executable = os.path.abspath('./C++/enc_txt')  # Absolute path to the executable
#     cpp_dir = os.path.dirname(cpp_encrypt_executable)  # Directory of the executable

#     # Move the file to the executable's directory
#     filename = secure_filename(uploaded_file.filename)
#     input_filepath = os.path.join(cpp_dir, filename)  # Copy file to the executable's directory
#     uploaded_file.save(input_filepath)

#     # Prepare arguments for the encryption command
#     args = [cpp_encrypt_executable, filename]

#     try:
#         logger.info(f"Executing encryption with args: {args}")
#         subprocess.run(args, check=True, cwd=cpp_dir)  # Set working directory to the executable's directory
#         encrypted_filename = f"encrypted_{filename}"
#         encrypted_filepath = os.path.join(cpp_dir, encrypted_filename)

#         if not os.path.exists(encrypted_filepath):
#             logger.error(f"Encrypted file not found: {encrypted_filepath}")
#             return render_template(
#                 'encrypt.html',
#                 message="Encryption failed. Encrypted file not found.",
#                 category="error",
#                 user_id=user_id
#             )

#         # Read the encrypted file
#         with open(encrypted_filepath, 'rb') as f:
#             encrypted_data = f.read()

#         # Clean up
#         os.remove(input_filepath)
#         os.remove(encrypted_filepath)

#         # Provide the encrypted file for download
#         return send_file(
#             io.BytesIO(encrypted_data),
#             as_attachment=True,
#             download_name=encrypted_filename
#         )
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Encryption subprocess error: {e}")
#         return render_template(
#             'encrypt.html',
#             message="An error occurred during encryption.",
#             category="error",
#             user_id=user_id
#         )
#     except Exception as e:
#         logger.error(f"Unexpected error: {e}")
#         return render_template(
#             'encrypt.html',
#             message="An unexpected error occurred.",
#             category="error",
#             user_id=user_id
#         )

@app.route('/encrypt')
def encrypt():
    # Check if the user is logged in
    if 'user_id' not in session:
        logger.info("No user_id in session. Redirecting to login.")
        return redirect(url_for('login'))

    # Fetch user details using `user_id` from the session
    user_id = session['user_id']
    user = users_collection.find_one({"user_id": user_id})

    # Check if the user exists and has completed enrollment
    if not user or not user.get('answers') or len(user['answers']) < 3:
        logger.info(f"User {user_id} not properly enrolled. Redirecting to secure dashboard.")
        return render_template(
            'secure_dashboard.html',
            message="Please complete enrollment first.",
            category="error",
            user_id=user_id
        )

    # Log the user's ID and proceed to encryption page
    logger.info(f"Rendering encrypt.html for user_id: {user_id}")

    # Render the encrypt page without any unrelated variables
    return render_template('encrypt.html', user_id=user_id)

@app.route('/process_encryption', methods=['POST'])
def process_encryption():
    if 'user_id' not in session:
        logger.info("No user_id in session. Redirecting to login.")
        return redirect(url_for('login'))

    # Fetch `user_id` from session
    user_id = session['user_id']
    uploaded_file = request.files.get('file', None)

    if not uploaded_file or uploaded_file.filename == '':
        return render_template(
            'encrypt.html',
            message="File is required.",
            category="error",
            user_id=user_id
        )

    if not allowed_file(uploaded_file.filename):
        return render_template(
            'encrypt.html',
            message="Unsupported file type.",
            category="error",
            user_id=user_id
        )

    # Fetch user data from MongoDB using `user_id`
    user = users_collection.find_one({"user_id": user_id})
    if not user:
        return render_template(
            'encrypt.html',
            message="User not found. Please enroll first.",
            category="error",
            user_id=user_id
        )

    answers = user.get('answers', {})
    if len(answers) < 3:
        return render_template(
            'encrypt.html',
            message="Insufficient transcriptions for encryption. Please ensure at least 3 questions are enrolled.",
            category="error",
            user_id=user_id
        )

    # Determine file type and select appropriate encryption executable and output pattern
    filename = secure_filename(uploaded_file.filename)
    file_ext = filename.rsplit('.', 1)[1].lower()

    encryption_info = ALLOWED_EXTENSIONS.get(file_ext)
    if not encryption_info:
        # This should not happen due to allowed_file check, but added for safety
        logger.error(f"Unsupported file extension encountered: {file_ext}")
        return render_template(
            'encrypt.html',
            message="Unsupported file type.",
            category="error",
            user_id=user_id
        )

    encryption_executable = encryption_info['executable']
    output_pattern = encryption_info['output_pattern']

    encryption_dir = os.path.dirname(encryption_executable)

    # Move the file to the executable's directory
    input_filepath = os.path.join(encryption_dir, filename)
    try:
        uploaded_file.save(input_filepath)
        logger.info(f"Uploaded file saved to {input_filepath}")
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return render_template(
            'encrypt.html',
            message="Failed to save the uploaded file.",
            category="error",
            user_id=user_id
        )

    # Prepare arguments for the encryption command
    args = [encryption_executable, filename]

    try:
        logger.info(f"Executing encryption with args: {args}")
        # Execute the encryption process
        result = subprocess.run(args, check=True, cwd=encryption_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info(f"Encryption subprocess stdout: {result.stdout}")
        logger.info(f"Encryption subprocess stderr: {result.stderr}")

        # Determine the expected encrypted file path
        if '{filename}' in output_pattern:
            encrypted_filename = output_pattern.format(filename=filename)
        else:
            encrypted_filename = output_pattern  # For audio files as per current behavior

        encrypted_filepath = os.path.join(encryption_dir, encrypted_filename)

        # Check if the encrypted file exists
        if not os.path.exists(encrypted_filepath):
            logger.error(f"Encrypted file not found: {encrypted_filepath}")
            return render_template(
                'encrypt.html',
                message="Encryption failed. Encrypted file not found.",
                category="error",
                user_id=user_id
            )

        # Read the encrypted file
        with open(encrypted_filepath, 'rb') as f:
            encrypted_data = f.read()
        logger.info(f"Encrypted file {encrypted_filename} read successfully.")

        # Clean up: Remove the original and encrypted files after reading
        try:
            os.remove(input_filepath)
            logger.info(f"Removed input file: {input_filepath}")
            os.remove(encrypted_filepath)
            logger.info(f"Removed encrypted file: {encrypted_filepath}")
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")

        # Determine the download filename based on file type
        if file_ext in {'txt', 'jpg', 'jpeg', 'png'}:
            download_filename = encrypted_filename
        elif file_ext in {'mp3', 'wav'}:
            # For audio, the encrypted file is a text file
            download_filename = encrypted_filename
        else:
            # Fallback in case of unexpected file type
            download_filename = encrypted_filename

        # Provide the encrypted file for download
        return send_file(
            io.BytesIO(encrypted_data),
            as_attachment=True,
            download_name=download_filename
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Encryption subprocess error: {e}")
        logger.error(f"Subprocess stdout: {e.stdout}")
        logger.error(f"Subprocess stderr: {e.stderr}")
        return render_template(
            'encrypt.html',
            message="An error occurred during encryption.",
            category="error",
            user_id=user_id
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return render_template(
            'encrypt.html',
            message="An unexpected error occurred.",
            category="error",
            user_id=user_id
        )


# ###############################################################################################################################
# ###############################################################################################################################

@app.route('/decrypt')
def decrypt():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('decrypt.html')
# ###############################################################################################################################
# ###############################################################################################################################

# @app.route('/process_decryption', methods=['POST'])
# def process_decryption():
#     if 'user_id' not in session:
#         logger.info("No user_id in session. Redirecting to login.")
#         return redirect(url_for('login'))
    
#     user_id = session['user_id']
#     uploaded_file = request.files.get('file', None)
#     output_file_type = request.form.get('file_type', '').lower()
    
#     # Validate uploaded file
#     if not uploaded_file or uploaded_file.filename == '':
#         logger.error("No file uploaded.")
#         return render_template('decrypt.html', message="File is required.", category="error", user_id=user_id)
    
#     filename = secure_filename(uploaded_file.filename)
#     if not allowed_file(filename):
#         logger.error(f"Unsupported encrypted file type: {filename}")
#         return render_template('decrypt.html', message="Unsupported encrypted file type. Please upload a .txt file.", category="error", user_id=user_id)
    
#     try:
#         temp_dir = tempfile.mkdtemp()
#         input_filepath = os.path.join(temp_dir, filename)
#         uploaded_file.save(input_filepath)
#         logger.info(f"Uploaded encrypted file saved at: {input_filepath}")
#     except Exception as e:
#         logger.error(f"Error saving uploaded file: {e}")
#         return render_template('decrypt.html', message="Failed to save the uploaded file.", category="error", user_id=user_id)
    
#     # Determine the decryption executable
#     decryption_executable = DECRYPTION_EXECUTABLES.get(output_file_type)
#     if not decryption_executable:
#         logger.error(f"No decryption executable configured for output type: {output_file_type}")
#         cleanup_temp_dir(temp_dir)
#         return render_template('decrypt.html', message="Invalid output file type selected.", category="error", user_id=user_id)
    
#     # Determine the expected output file name based on the decryption type
#     if output_file_type in {'mp3', 'wav'}:
#         # For audio decryption, the output file is a fixed name
#         output_filename = "reconstructed_audio.wav"
#     else:
#         # For text or images, derive the output file name dynamically
#         output_filename = os.path.basename(filename).rsplit('.', 1)[0] + "_decrypted.txt"
#     decrypted_filepath = os.path.join(temp_dir, output_filename)
    
#     # Execute the decryption subprocess
#     try:
#         logger.info(f"Executing decryption with args: [{decryption_executable}, {input_filepath}]")
#         subprocess.run(
#             [decryption_executable, input_filepath],
#             check=True,
#             cwd=temp_dir,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True
#         )
#         logger.info("Decryption subprocess completed.")
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Decryption subprocess error: {e}")
#         logger.error(f"Subprocess stdout: {e.stdout}")
#         logger.error(f"Subprocess stderr: {e.stderr}")
#         cleanup_temp_dir(temp_dir)
#         return render_template('decrypt.html', message="An error occurred during decryption.", category="error", user_id=user_id)
#     except Exception as e:
#         logger.error(f"Unexpected error during decryption: {e}")
#         cleanup_temp_dir(temp_dir)
#         return render_template('decrypt.html', message="An unexpected error occurred during decryption.", category="error", user_id=user_id)
    
#     # Verify that the decrypted file exists
#     if not os.path.exists(decrypted_filepath):
#         logger.error(f"Decrypted file not found: {decrypted_filepath}")
#         cleanup_temp_dir(temp_dir)
#         return render_template('decrypt.html', message="Decryption failed. Decrypted file not found.", category="error", user_id=user_id)
    
#     try:
#         with open(decrypted_filepath, 'rb') as f:
#             decrypted_data = f.read()
#         logger.info(f"Decrypted file {decrypted_filepath} read successfully.")
#         cleanup_temp_dir(temp_dir)
#         return send_file(
#             io.BytesIO(decrypted_data),
#             as_attachment=True,
#             download_name=f"decrypted_{filename.rsplit('.', 1)[0]}.{output_file_type}"
#         )
#     except Exception as e:
#         logger.error(f"Error serving decrypted file: {e}")
#         cleanup_temp_dir(temp_dir)
#         return render_template('decrypt.html', message="Failed to serve the decrypted file.", category="error", user_id=user_id)
@app.route('/process_decryption', methods=['POST'])
def process_decryption():
    if 'user_id' not in session:
        logger.info("No user_id in session. Redirecting to login.")
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    uploaded_file = request.files.get('file', None)
    output_file_type = request.form.get('file_type', '').lower()
    
    # Validate uploaded file
    if not uploaded_file or uploaded_file.filename == '':
        logger.error("No file uploaded.")
        return render_template('decrypt.html', message="File is required.", category="error", user_id=user_id)
    
    filename = secure_filename(uploaded_file.filename)
    if not allowed_file(filename):
        logger.error(f"Unsupported encrypted file type: {filename}")
        return render_template('decrypt.html', message="Unsupported encrypted file type. Please upload a valid file.", category="error", user_id=user_id)
    
    try:
        temp_dir = tempfile.mkdtemp()
        input_filepath = os.path.join(temp_dir, filename)
        uploaded_file.save(input_filepath)
        logger.info(f"Uploaded encrypted file saved at: {input_filepath}")
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        return render_template('decrypt.html', message="Failed to save the uploaded file.", category="error", user_id=user_id)
    
    # Determine the decryption executable
    decryption_executable = DECRYPTION_EXECUTABLES.get(output_file_type)
    if not decryption_executable:
        logger.error(f"No decryption executable configured for output type: {output_file_type}")
        cleanup_temp_dir(temp_dir)
        return render_template('decrypt.html', message="Invalid output file type selected.", category="error", user_id=user_id)
    
    # Set the expected output file name based on the decryption type
    if output_file_type in {'jpg', 'jpeg', 'png'}:
        # For images, the output file is fixed as 'decrypted_image.png'
        output_filename = "decrypted_image.png"
    elif output_file_type in {'mp3', 'wav'}:
        # For audio, the output file is fixed as 'reconstructed_audio.wav'
        output_filename = "reconstructed_audio.wav"
    elif output_file_type in {'txt'}:
        # For text, derive the output file name dynamically
        output_filename = os.path.basename(filename).rsplit('.', 1)[0] + "_decrypted.txt"
    else:
        # Default behavior for other file types
        output_filename = f"decrypted_{filename}"
    decrypted_filepath = os.path.join(temp_dir, output_filename)
    
    # Execute the decryption subprocess
    try:
        logger.info(f"Executing decryption with args: [{decryption_executable}, {input_filepath}]")
        subprocess.run(
            [decryption_executable, input_filepath],
            check=True,
            cwd=temp_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("Decryption subprocess completed.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Decryption subprocess error: {e}")
        logger.error(f"Subprocess stdout: {e.stdout}")
        logger.error(f"Subprocess stderr: {e.stderr}")
        cleanup_temp_dir(temp_dir)
        return render_template('decrypt.html', message="An error occurred during decryption.", category="error", user_id=user_id)
    except Exception as e:
        logger.error(f"Unexpected error during decryption: {e}")
        cleanup_temp_dir(temp_dir)
        return render_template('decrypt.html', message="An unexpected error occurred during decryption.", category="error", user_id=user_id)
    
    # Verify that the decrypted file exists
    if not os.path.exists(decrypted_filepath):
        logger.error(f"Decrypted file not found: {decrypted_filepath}")
        cleanup_temp_dir(temp_dir)
        return render_template('decrypt.html', message="Decryption failed. Decrypted file not found.", category="error", user_id=user_id)
    
    try:
        with open(decrypted_filepath, 'rb') as f:
            decrypted_data = f.read()
        logger.info(f"Decrypted file {decrypted_filepath} read successfully.")
        cleanup_temp_dir(temp_dir)
        return send_file(
            io.BytesIO(decrypted_data),
            as_attachment=True,
            download_name=f"decrypted_{filename.rsplit('.', 1)[0]}.{output_file_type}"
        )
    except Exception as e:
        logger.error(f"Error serving decrypted file: {e}")
        cleanup_temp_dir(temp_dir)
        return render_template('decrypt.html', message="Failed to serve the decrypted file.", category="error", user_id=user_id)

# ###############################################################################################################################
# ###############################################################################################################################
# ###############################################################################################################################
# ###############################################################################################################################

if __name__ == '__main__':
    app.run(debug=True)
