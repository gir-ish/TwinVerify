import streamlit as st
from scipy.io.wavfile import write, read
from speechbrain.pretrained import SpeakerRecognition
import torch
import numpy as np
from io import BytesIO
import os
from collections import defaultdict
import sounddevice as sd

# Initialize the speaker recognition model
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")

# Global user data structure
user_data = defaultdict(lambda: {'embeddings': [], 'memory_buffer': [], 'verification_count': 0})
SIMILARITY_THRESHOLD_ENROLL = 0.50
SIMILARITY_THRESHOLD_VERIFY = 0.65


# Utility Functions
def record_audio(duration=5, fs=16000):
    """Record audio for a specified duration."""
    st.info("Recording... Please speak.")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.success("Recording completed.")
    return audio_data.squeeze()


def compute_embedding(audio_data, fs=16000):
    """Compute the embedding for given audio data."""
    audio_data = (audio_data * 32767).astype(np.int16)  # Convert to 16-bit PCM
    byte_io = BytesIO()
    write(byte_io, fs, audio_data)
    byte_io.seek(0)
    audio_tensor = torch.tensor(read(byte_io)[1], dtype=torch.float32).unsqueeze(0)
    return verification.encode_batch(audio_tensor)


def compare_embeddings(embedding1, embedding2):
    """Compare two embeddings and return the similarity score."""
    return verification.similarity(embedding1, embedding2).item()


def update_embedding(old_embedding, new_embedding, alpha=0.5):
    """Update embedding using weighted averaging and normalization."""
    updated_embedding = alpha * old_embedding + (1 - alpha) * new_embedding
    return updated_embedding / torch.norm(updated_embedding)


def save_user_data(user_id):
    """Save user data to disk."""
    user_dir = f'vault_data/{user_id}'
    os.makedirs(user_dir, exist_ok=True)
    for idx, embedding in enumerate(user_data[user_id]['embeddings']):
        torch.save(embedding, f'{user_dir}/embedding_{idx + 1}.pt')
    with open(f'{user_dir}/verification_count.txt', 'w') as f:
        f.write(str(user_data[user_id]['verification_count']))


def load_user_data(user_id):
    """Load user data from disk."""
    user_dir = f'vault_data/{user_id}'
    if os.path.exists(user_dir):
        embeddings = [
            torch.load(os.path.join(user_dir, file))
            for file in sorted(os.listdir(user_dir)) if file.endswith('.pt')
        ]
        verification_count = 0
        verification_count_file = os.path.join(user_dir, 'verification_count.txt')
        if os.path.exists(verification_count_file):
            with open(verification_count_file, 'r') as f:
                verification_count = int(f.read())
        user_data[user_id] = {
            'embeddings': embeddings,
            'memory_buffer': [],
            'verification_count': verification_count
        }


def remove_user(user_id):
    """Remove a user and their data."""
    user_dir = f'vault_data/{user_id}'
    if os.path.exists(user_dir):
        for file in os.listdir(user_dir):
            os.remove(os.path.join(user_dir, file))
        os.rmdir(user_dir)
    if user_id in user_data:
        del user_data[user_id]


# Streamlit App UI
st.title("Voice Authentication System")

# Sidebar for navigation
menu = st.sidebar.radio(
    "Navigation",
    ["Enroll a New User", "Verify a User", "Manage Users"]
)

# Enrollment Page
if menu == "Enroll a New User":
    st.header("Enroll a New User")
    user_id = st.text_input("Enter a unique User ID:")

    if st.button("Start Enrollment"):
        if not user_id:
            st.error("User ID cannot be empty!")
        elif user_id in user_data:
            st.error(f"User {user_id} is already enrolled!")
        else:
            st.session_state["enrollment_audio"] = []
            for i in range(3):
                st.write(f"Recording segment {i + 1}...")
                audio_data = record_audio()
                embedding = compute_embedding(audio_data)

                if i > 0:
                    similarity = compare_embeddings(
                        st.session_state["enrollment_audio"][-1], embedding
                    )
                    if similarity < SIMILARITY_THRESHOLD_ENROLL:
                        st.warning(
                            f"Segment {i + 1} has low similarity with the previous. Please re-record."
                        )
                        continue
                st.session_state["enrollment_audio"].append(embedding)

            user_data[user_id] = {
                "embeddings": st.session_state["enrollment_audio"],
                "memory_buffer": [],
                "verification_count": 0,
            }
            save_user_data(user_id)
            st.success(f"User {user_id} enrolled successfully with 3 segments!")

# Verification Page
elif menu == "Verify a User":
    st.header("Verify a User")
    user_id = st.text_input("Enter the User ID to verify:")

    if st.button("Start Verification"):
        if not user_id:
            st.error("User ID cannot be empty!")
        elif user_id not in user_data:
            st.error(f"User {user_id} is not enrolled!")
        else:
            st.write("Recording verification audio...")
            verification_audio = record_audio()
            verification_embedding = compute_embedding(verification_audio)

            max_similarity, best_match_idx = 0, -1
            for idx, embedding in enumerate(user_data[user_id]["embeddings"]):
                similarity = compare_embeddings(embedding, verification_embedding)
                if similarity > max_similarity:
                    max_similarity, best_match_idx = similarity, idx

            if max_similarity > SIMILARITY_THRESHOLD_VERIFY:
                st.success(f"Verification successful with similarity {max_similarity:.2f}.")
                updated_embedding = update_embedding(
                    user_data[user_id]["embeddings"][best_match_idx],
                    verification_embedding,
                )
                user_data[user_id]["embeddings"][best_match_idx] = updated_embedding
                save_user_data(user_id)
            else:
                st.error(f"Verification failed. Maximum similarity: {max_similarity:.2f}")

# Manage Users Page
elif menu == "Manage Users":
    st.header("Manage Users")
    users = list(user_data.keys())

    if not users:
        st.warning("No users enrolled yet!")
    else:
        st.write("Enrolled Users:")
        for user in users:
            st.write(f"- {user}")

        user_id = st.text_input("Enter User ID to remove:")
        if st.button("Remove User"):
            if not user_id:
                st.error("User ID cannot be empty!")
            elif user_id not in user_data:
                st.error(f"User {user_id} does not exist!")
            else:
                remove_user(user_id)
                st.success(f"User {user_id} has been removed.")
