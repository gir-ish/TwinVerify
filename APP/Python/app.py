import sounddevice as sd
from scipy.io.wavfile import write, read
from speechbrain.pretrained import SpeakerRecognition
import torch
import os
import numpy as np
from io import BytesIO
from collections import defaultdict
import warnings
import time

warnings.filterwarnings("ignore")

# Initialize the speaker recognition model
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")

# Global user data structure
user_data = defaultdict(lambda: {'embeddings': [], 'memory_buffer': [], 'verification_count': 0})

# Thresholds
SIMILARITY_THRESHOLD_ENROLL = 0.50
SIMILARITY_THRESHOLD_VERIFY = 0.65

def record_audio(duration=5, fs=16000):
    """Record audio for a specified duration."""
    print("Recording... Please speak.")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording completed.")
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
        # Handle missing verification count file
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
        print(f"User {user_id} loaded with {len(embeddings)} embeddings and verification count {verification_count}.")
    else:
        print(f"User directory for {user_id} does not exist. Skipping...")

def enroll_user(user_id):
    """Enroll a user by recording and saving multiple audio segments."""
    if user_id in user_data:
        print(f"User {user_id} already exists!")
        return

    print(f"Starting enrollment for {user_id}.")
    embeddings = []
    for i in range(3):  # Reduced to 3 recordings for efficiency
        audio_data = record_audio()
        embedding = compute_embedding(audio_data)
        if embeddings:
            similarity = compare_embeddings(embeddings[-1], embedding)
            if similarity < SIMILARITY_THRESHOLD_ENROLL:
                print(f"Low similarity with previous segment ({similarity:.2f}). Please re-record.")
                continue
        embeddings.append(embedding)
        print(f"Segment {i + 1} recorded.")

    user_data[user_id] = {'embeddings': embeddings, 'memory_buffer': [], 'verification_count': 0}
    save_user_data(user_id)
    print(f"User {user_id} enrolled successfully with {len(embeddings)} segments.")

def verify_user(user_id):
    """Verify a user by comparing their recording with saved embeddings."""
    if user_id not in user_data:
        print(f"User {user_id} not found!")
        return

    print(f"Recording verification audio for {user_id}...")
    verification_audio = record_audio()
    verification_embedding = compute_embedding(verification_audio)

    max_similarity, best_match_idx = 0, -1
    for idx, embedding in enumerate(user_data[user_id]['embeddings']):
        similarity = compare_embeddings(embedding, verification_embedding)
        if similarity > max_similarity:
            max_similarity, best_match_idx = similarity, idx

    if max_similarity > SIMILARITY_THRESHOLD_VERIFY:
        print(f"Verification successful with similarity {max_similarity:.2f}. Updating embedding...")
        updated_embedding = update_embedding(user_data[user_id]['embeddings'][best_match_idx], verification_embedding)
        user_data[user_id]['embeddings'][best_match_idx] = updated_embedding
        save_user_data(user_id)
    else:
        print(f"Verification failed. Maximum similarity: {max_similarity:.2f}.")

def list_users():
    """List all enrolled users."""
    users = list(user_data.keys())
    print("Enrolled Users:" if users else "No users found.")
    for user in users:
        print(f"- {user}")

def remove_user(user_id):
    """Remove a user and their data from disk."""
    if user_id in user_data:
        del user_data[user_id]
        user_dir = f'vault_data/{user_id}'
        if os.path.exists(user_dir):
            for file in os.listdir(user_dir):
                os.remove(os.path.join(user_dir, file))
            os.rmdir(user_dir)
        print(f"User {user_id} removed.")
    else:
        print(f"User {user_id} not found.")

def main_menu():
    """Main menu for the application."""
    while True:
        print("\n--- Voice Authentication System ---")
        print("1. Enroll a new user")
        print("2. Verify a user")
        print("3. List enrolled users")
        print("4. Remove a user")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            user_id = input("Enter user ID: ")
            enroll_user(user_id)
        elif choice == '2':
            user_id = input("Enter user ID to verify: ")
            verify_user(user_id)
        elif choice == '3':
            list_users()
        elif choice == '4':
            user_id = input("Enter user ID to remove: ")
            remove_user(user_id)
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

# Load existing users on startup
if __name__ == "__main__":
    if not os.path.exists('vault_data'):
        os.makedirs('vault_data')
    for user_dir in os.listdir('vault_data'):
        load_user_data(user_dir)
    main_menu()
