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

# User data structure with embeddings, memory buffer, and verification count
user_data = defaultdict(lambda: {'embeddings': [], 'memory_buffer': [], 'verification_count': 0})

SIMILARITY_THRESHOLD_ENROLL = 0.50  # Threshold for enrollment recordings
SIMILARITY_THRESHOLD_VERIFY = 0.65  # Threshold for verification

def record_audio_with_timer(duration=5, fs=16000):
    """Record audio and display countdown timer while recording."""
    print("Recording will start...")
    audio_data = np.empty((int(duration * fs), 1), dtype=np.float32)  # Pre-allocate array
    stream = sd.InputStream(samplerate=fs, channels=1)

    with stream:
        for i in range(duration, 0, -1):  # Countdown timer
            print(f"Recording... {i} seconds remaining")
            sd.sleep(1000)  # Wait for 1 second before updating the countdown
        audio_data = stream.read(int(duration * fs))[0]

    print("Recording completed.")
    return audio_data

def compute_embedding(audio_data):
    """Helper function to compute the embedding of the given audio."""
    byte_io = BytesIO()
    write(byte_io, 16000, audio_data)  # Store audio in memory (not on disk)
    byte_io.seek(0)  # Move pointer to the start

    audio_array = np.array(read(byte_io)[1])
    if audio_array.ndim > 1:  # Ensure it's mono
        audio_array = audio_array[:, 0]

    audio_tensor = torch.tensor(audio_array).unsqueeze(0)  # 1D tensor: [1, num_samples]
    embedding = verification.encode_batch(audio_tensor)
    return embedding

def compare_embeddings(embedding1, embedding2):
    """Compare two embeddings and return the similarity score."""
    similarity = verification.similarity(embedding1, embedding2)
    return similarity.item()  # Convert tensor to Python float

def update_embedding_incrementally(old_embedding, new_embedding, alpha=0.5):
    """Update embedding by combining old and new embeddings."""
    updated_embedding = alpha * old_embedding + (1 - alpha) * new_embedding
    # Normalize the updated embedding
    updated_embedding = updated_embedding / torch.norm(updated_embedding)
    return updated_embedding

def enroll_user(user_id):
    """Enroll a user by recording multiple segments, allowing re-recording."""
    if user_id in user_data:
        print(f"User {user_id} already exists!")
        return

    user_dir = f'vault_data/{user_id}'
    os.makedirs(user_dir, exist_ok=True)

    embeddings = []  # To store individual embeddings for the user

    for i in range(10):
        print(f"\nRecording segment {i + 1} for {user_id}...")
        segment = record_audio_with_timer(duration=5)

        # Ask user if they want to keep or re-record the audio
        choice = input("Do you want to re-record this segment (y/n)? ").lower()
        if choice == 'y':
            segment = record_audio_with_timer(duration=5)  # Re-record

        embedding = compute_embedding(segment)
        embeddings.append(embedding)
        print(f"Segment {i + 1} recorded and saved.")

        # If it's not the first segment, check similarity with the previous one
        if i > 0:
            similarity = compare_embeddings(embeddings[i - 1], embeddings[i])
            print(f"Similarity with the previous segment: {similarity:.2f}")
            
            # Check if similarity is below the enrollment threshold
            if similarity < SIMILARITY_THRESHOLD_ENROLL:
                print(f"Similarity {similarity:.2f} is below the threshold {SIMILARITY_THRESHOLD_ENROLL}.")
                choice = input("Do you want to re-record this segment (y/n)? ").lower()
                if choice == 'y':
                    segment = record_audio_with_timer(duration=5)  # Re-record
                    embedding = compute_embedding(segment)
                    embeddings[i] = embedding  # Update the embedding
                    print(f"Segment {i + 1} updated.")

    # Save all embeddings separately
    for idx, embedding in enumerate(embeddings):
        torch.save(embedding, f'{user_dir}/{user_id}_embedding_{idx + 1}.pt')

    # Save verification count
    with open(f'{user_dir}/verification_count.txt', 'w') as f:
        f.write('0')

    # Initialize user data
    user_data[user_id] = {
        'embeddings': embeddings,
        'memory_buffer': [],  # Initialize memory buffer empty
        'verification_count': 0
    }
    print(f"User {user_id} enrolled successfully with {len(embeddings)} segments.")

def verify_user(user_id):
    """Verify a user by comparing their new recording with saved embeddings."""
    if user_id not in user_data:
        print(f"User {user_id} not found!")
        return

    # Record a new sample for verification
    print(f"\nRecording verification audio for {user_id}...")
    verification_segment = record_audio_with_timer(duration=5)
    verification_embedding = compute_embedding(verification_segment)

    # Compare with each saved embedding and find the highest similarity
    max_similarity = 0
    best_match_idx = -1

    for idx, embedding in enumerate(user_data[user_id]['embeddings']):
        similarity = compare_embeddings(embedding, verification_embedding)
        print(f"Similarity with embedding {idx + 1}: {similarity:.2f}")

        if similarity > max_similarity:
            max_similarity = similarity
            best_match_idx = idx

    print(f"\nBest match similarity: {max_similarity:.2f} with embedding {best_match_idx + 1}")

    if max_similarity > SIMILARITY_THRESHOLD_VERIFY:
        print(f"Verification successful for {user_id} with embedding {best_match_idx + 1}.")

        # Update the matching embedding using Continual Learning
        old_embedding = user_data[user_id]['embeddings'][best_match_idx]
        updated_embedding = update_embedding_incrementally(old_embedding, verification_embedding)
        user_data[user_id]['embeddings'][best_match_idx] = updated_embedding
        torch.save(updated_embedding, f'vault_data/{user_id}/{user_id}_embedding_{best_match_idx + 1}.pt')
        print(f"Embedding {best_match_idx + 1} updated with Continual Learning.")

        # Increment verification count
        user_data[user_id]['verification_count'] += 1

        # Save updated verification count
        user_dir = f'vault_data/{user_id}'
        with open(f'{user_dir}/verification_count.txt', 'w') as f:
            f.write(str(user_data[user_id]['verification_count']))

        # Update memory buffer
        user_data[user_id]['memory_buffer'].append((best_match_idx, verification_embedding))
        # Limit memory buffer size if necessary
        MAX_MEMORY_SIZE = 50
        if len(user_data[user_id]['memory_buffer']) > MAX_MEMORY_SIZE:
            user_data[user_id]['memory_buffer'].pop(0)  # Remove oldest

        # Perform Rehearsal-based Learning every 10 successful verifications
        if user_data[user_id]['verification_count'] % 10 == 0:
            print("Performing Rehearsal-based Learning...")
            perform_rehearsal_learning(user_id)
            print(f"User {user_id}'s embeddings updated with Rehearsal-based Learning.")
    else:
        print(f"Verification failed.")

def perform_rehearsal_learning(user_id):
    """Update embeddings using Rehearsal-based Learning without averaging."""
    # For each embedding, collect all corresponding updates from memory buffer
    memory_buffer = user_data[user_id]['memory_buffer']
    embedding_updates = defaultdict(list)

    for idx, emb in memory_buffer:
        embedding_updates[idx].append(emb)

    # Update each embedding individually
    for idx, updates in embedding_updates.items():
        # Original embedding
        original_embedding = user_data[user_id]['embeddings'][idx]

        # Update embedding incrementally with all updates
        for update_embedding in updates:
            original_embedding = update_embedding_incrementally(original_embedding, update_embedding)

        # Save updated embedding
        user_data[user_id]['embeddings'][idx] = original_embedding
        torch.save(original_embedding, f'vault_data/{user_id}/{user_id}_embedding_{idx + 1}.pt')

    # Clear memory buffer after rehearsal learning
    user_data[user_id]['memory_buffer'] = []

def load_existing_users():
    """Load existing users from the vault directory on application start."""
    if not os.path.exists('vault_data'):
        os.makedirs('vault_data')

    # Check each subdirectory in the vault_data folder
    for user_id in os.listdir('vault_data'):
        user_dir = f'vault_data/{user_id}'
        if os.path.isdir(user_dir):
            # Load the embeddings for each user
            embeddings = []
            files = sorted([f for f in os.listdir(user_dir) if f.endswith(".pt")])
            for file in files:
                embedding = torch.load(os.path.join(user_dir, file))
                embeddings.append(embedding)

            # Load verification count
            verification_count = 0
            verification_count_file = os.path.join(user_dir, 'verification_count.txt')
            if os.path.exists(verification_count_file):
                with open(verification_count_file, 'r') as f:
                    verification_count = int(f.read())

            if embeddings:
                user_data[user_id] = {
                    'embeddings': embeddings,
                    'memory_buffer': [],
                    'verification_count': verification_count
                }
                print(f"User {user_id} loaded with {len(embeddings)} embeddings and verification count {verification_count}.")

def remove_user(user_id):
    """Remove a user and their vault."""
    if user_id in user_data:
        del user_data[user_id]
        user_dir = f'vault_data/{user_id}'
        if os.path.exists(user_dir):
            for file in os.listdir(user_dir):
                os.remove(os.path.join(user_dir, file))
            os.rmdir(user_dir)
        print(f"User {user_id} and their data have been removed.")
    else:
        print(f"User {user_id} not found.")

# Main application menu
def menu():
    load_existing_users()  # Load existing users on start
    while True:
        print("\n--- Voice Authentication System ---")
        print("1. Enroll a new user")
        print("2. Verify a user")
        print("3. Load existing users")
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
            load_existing_users()
        elif choice == '4':
            user_id = input("Enter user ID to remove: ")
            remove_user(user_id)
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.")

# Run the menu
if __name__ == "__main__":
    menu()
