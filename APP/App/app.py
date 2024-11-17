import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write, read
from speechbrain.pretrained import SpeakerRecognition
import torch
import os
import numpy as np
from io import BytesIO
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# Initialize SpeakerRecognition model
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")

# User data storage
user_data = defaultdict(list)

# Thresholds
SIMILARITY_THRESHOLD_ENROLL = 0.50
SIMILARITY_THRESHOLD_VERIFY = 0.70

# Streamlit page setup
st.title("Voice Authentication System")

def record_audio(duration=5, fs=16000):
    """Record audio using the microphone for a specified duration."""
    audio_data = np.empty((int(duration * fs), 1), dtype=np.float32)
    stream = sd.InputStream(samplerate=fs, channels=1)
    
    with stream:
        st.write("Recording... Please wait")
        audio_data = stream.read(int(duration * fs))[0]

    st.success("Recording completed!")
    return audio_data

def compute_embedding(audio_data):
    """Helper function to compute the embedding of the given audio."""
    byte_io = BytesIO()
    write(byte_io, 16000, audio_data)
    byte_io.seek(0)

    audio_array = np.array(read(byte_io)[1])
    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    audio_tensor = torch.tensor(audio_array).unsqueeze(0)
    embedding = verification.encode_batch(audio_tensor)
    return embedding

def compare_embeddings(embedding1, embedding2):
    """Compare two embeddings and return the similarity score."""
    similarity = verification.similarity(embedding1, embedding2)
    return similarity.item()

def enroll_user(user_id):
    """Enroll a user by recording multiple segments."""
    if user_id in user_data:
        st.error(f"User {user_id} already exists!")
        return

    user_dir = f'vault_data/{user_id}'
    os.makedirs(user_dir, exist_ok=True)

    embeddings = []

    for i in range(5):
        st.write(f"\nRecording segment {i + 1} for {user_id}...")
        segment = record_audio(duration=5)

        # Ask if the user wants to re-record this segment
        re_record = st.radio(f"Do you want to re-record segment {i + 1}?", options=["No", "Yes"])

        while re_record == "Yes":
            segment = record_audio(duration=5)
            re_record = st.radio(f"Do you want to re-record segment {i + 1}?", options=["No", "Yes"])

        embedding = compute_embedding(segment)
        embeddings.append(embedding)

        if i > 0:
            similarity = compare_embeddings(embeddings[i - 1], embeddings[i])
            st.write(f"Similarity with previous segment: {similarity:.2f}")
            if similarity < SIMILARITY_THRESHOLD_ENROLL:
                st.warning(f"Low similarity ({similarity:.2f}) detected. You may want to re-record.")
                re_record = st.radio(f"Do you want to re-record segment {i + 1} due to low similarity?", options=["No", "Yes"])
                while re_record == "Yes":
                    segment = record_audio(duration=5)
                    embedding = compute_embedding(segment)
                    embeddings[i] = embedding
                    similarity = compare_embeddings(embeddings[i - 1], embeddings[i])
                    st.write(f"Similarity with previous segment: {similarity:.2f}")
                    re_record = st.radio(f"Do you want to re-record segment {i + 1}?", options=["No", "Yes"])

    for idx, embedding in enumerate(embeddings):
        torch.save(embedding, f'{user_dir}/{user_id}_embedding_{idx + 1}.pt')

    user_data[user_id] = embeddings
    st.success(f"User {user_id} enrolled with {len(embeddings)} segments!")

def verify_user(user_id):
    """Verify a user by comparing their new recording with saved embeddings."""
    if user_id not in user_data:
        st.error(f"User {user_id} not found!")
        return

    st.write(f"Recording verification audio for {user_id}...")
    verification_segment = record_audio(duration=5)
    verification_embedding = compute_embedding(verification_segment)

    max_similarity = 0
    best_match_idx = -1

    for idx, embedding in enumerate(user_data[user_id]):
        similarity = compare_embeddings(embedding, verification_embedding)
        st.write(f"Similarity with embedding {idx + 1}: {similarity:.2f}")

        if similarity > max_similarity:
            max_similarity = similarity
            best_match_idx = idx

    st.write(f"Best match similarity: {max_similarity:.2f} with embedding {best_match_idx + 1}")

    if max_similarity > SIMILARITY_THRESHOLD_VERIFY:
        st.success(f"Verification successful for {user_id} with embedding {best_match_idx + 1}.")
        user_data[user_id][best_match_idx] = verification_embedding
        torch.save(verification_embedding, f'vault_data/{user_id}/{user_id}_embedding_{best_match_idx + 1}.pt')
        st.success(f"Embedding {best_match_idx + 1} updated.")
    else:
        st.error(f"Verification failed. Similarity did not meet the threshold ({SIMILARITY_THRESHOLD_VERIFY}).")

def load_existing_users():
    """Load existing users from the vault_data directory."""
    if not os.path.exists('vault_data'):
        os.makedirs('vault_data')

    for user_id in os.listdir('vault_data'):
        user_dir = f'vault_data/{user_id}'
        if os.path.isdir(user_dir):
            embeddings = []
            for file in os.listdir(user_dir):
                if file.endswith(".pt"):
                    embeddings.append(torch.load(os.path.join(user_dir, file)))
            if embeddings:
                user_data[user_id] = embeddings
                st.success(f"User {user_id} loaded with {len(embeddings)} embeddings.")

def remove_user(user_id):
    """Remove a user and their data."""
    if user_id in user_data:
        del user_data[user_id]
        user_dir = f'vault_data/{user_id}'
        if os.path.exists(user_dir):
            for file in os.listdir(user_dir):
                os.remove(os.path.join(user_dir, file))
            os.rmdir(user_dir)
        st.success(f"User {user_id} and their data have been removed.")

# Load existing users automatically on app startup
load_existing_users()

# Streamlit menu
option = st.selectbox(
    "Select an option", 
    ("Enroll a new user", "Verify a user", "Remove a user")
)

if option == "Enroll a new user":
    user_id = st.text_input("Enter user ID to enroll")
    if st.button("Enroll"):
        enroll_user(user_id)

elif option == "Verify a user":
    user_id = st.text_input("Enter user ID to verify")
    if st.button("Verify"):
        verify_user(user_id)

elif option == "Remove a user":
    user_id = st.text_input("Enter user ID to remove")
    if st.button("Remove"):
        remove_user(user_id)
