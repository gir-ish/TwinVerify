# Install the required packages before running the app
# pip install streamlit sounddevice speechbrain torch

import streamlit as st
import sounddevice as sd
from speechbrain.pretrained import SpeakerRecognition
import torch
import os
import numpy as np
from scipy.io.wavfile import write, read
from io import BytesIO
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# Initialize the speaker recognition model
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp"
)

# Initialize user data in session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = defaultdict(
        lambda: {'embeddings': [], 'memory_buffer': [], 'verification_count': 0}
    )
user_data = st.session_state.user_data

SIMILARITY_THRESHOLD_ENROLL = 0.50  # Threshold for enrollment recordings
SIMILARITY_THRESHOLD_VERIFY = 0.6 # Threshold for verification

def record_audio_with_timer(duration=5, fs=16000):
    """Record audio."""
    st.write("Recording...")
    # Record audio data using sounddevice
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    st.write("Recording completed.")
    return audio_data.flatten()

def compute_embedding(audio_data):
    """Helper function to compute the embedding of the given audio."""
    # Convert audio data to 16-bit PCM WAV format
    byte_io = BytesIO()
    # Ensure the audio data is in int16 format
    audio_int16 = (audio_data * 32767).astype(np.int16)
    write(byte_io, 16000, audio_int16)  # Store audio in memory (not on disk)
    byte_io.seek(0)  # Move pointer to the start

    audio_array = np.array(read(byte_io)[1])
    if audio_array.ndim > 1:  # Ensure it's mono
        audio_array = audio_array[:, 0]

    audio_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
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
            original_embedding = update_embedding_incrementally(
                original_embedding, update_embedding
            )

        # Save updated embedding
        user_data[user_id]['embeddings'][idx] = original_embedding
        torch.save(
            original_embedding,
            f'vault_data/{user_id}/{user_id}_embedding_{idx + 1}.pt',
        )

    # Clear memory buffer after rehearsal learning
    user_data[user_id]['memory_buffer'] = []

def load_existing_users():
    """Load existing users from the vault directory on application start."""
    if 'users_loaded' in st.session_state:
        return

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
                    'verification_count': verification_count,
                }
                st.write(
                    f"User {user_id} loaded with {len(embeddings)} embeddings and verification count {verification_count}."
                )

    st.session_state.users_loaded = True

def enroll_user():
    """Enroll a user by recording multiple segments, allowing re-recording."""
    st.write("Enroll a new user")

    if 'enroll_user_id' not in st.session_state:
        st.session_state.enroll_user_id = ''

    user_id = st.text_input("Enter user ID", st.session_state.enroll_user_id)
    st.session_state.enroll_user_id = user_id

    if user_id:
        if user_id in user_data:
            st.warning(f"User {user_id} already exists!")
            return
        else:
            if 'enroll_embeddings' not in st.session_state:
                st.session_state.enroll_embeddings = []
            if 'enroll_segment' not in st.session_state:
                st.session_state.enroll_segment = 1

            embeddings = st.session_state.enroll_embeddings

            if st.session_state.enroll_segment <= 10:
                st.write(f"\nRecording segment {st.session_state.enroll_segment} for {user_id}...")

                # Add an "Enter" button to start recording
                if st.button("Enter", key=f"enter_{st.session_state.enroll_segment}"):
                    audio_data = record_audio_with_timer(duration=5)
                    # Store audio_data in session_state
                    st.session_state['enroll_audio_data'] = audio_data
                    # Display audio player
                    st.audio(audio_data, format='audio/wav', sample_rate=16000)

                # Check if audio_data is in session_state
                if 'enroll_audio_data' in st.session_state:
                    audio_data = st.session_state['enroll_audio_data']
                    # Display audio player
                    st.audio(audio_data, format='audio/wav', sample_rate=16000)

                    # Options to keep or re-record
                    if st.button("Keep this recording", key=f"keep_{st.session_state.enroll_segment}"):
                        embedding = compute_embedding(audio_data)
                        embeddings.append(embedding)
                        st.session_state.enroll_embeddings = embeddings

                        # Clear the stored audio_data
                        del st.session_state['enroll_audio_data']

                        # Similarity check
                        if st.session_state.enroll_segment > 1:
                            similarity = compare_embeddings(embeddings[-2], embeddings[-1])
                            st.write(f"Similarity with the previous segment: {similarity:.2f}")

                            if similarity < SIMILARITY_THRESHOLD_ENROLL:
                                st.write(f"Similarity {similarity:.2f} is below the threshold {SIMILARITY_THRESHOLD_ENROLL}.")
                                if st.button("Proceed anyway", key=f"proceed_{st.session_state.enroll_segment}"):
                                    st.session_state.enroll_segment += 1
                                    st.success(f"Segment {st.session_state.enroll_segment - 1} recorded and saved.")
                                    st.experimental_rerun()
                                else:
                                    st.experimental_rerun()
                            else:
                                st.session_state.enroll_segment += 1
                                st.success(f"Segment {st.session_state.enroll_segment - 1} recorded and saved.")
                                st.experimental_rerun()
                        else:
                            st.session_state.enroll_segment += 1
                            st.success(f"Segment {st.session_state.enroll_segment - 1} recorded and saved.")
                            st.experimental_rerun()
                    elif st.button("Re-record", key=f"rerecord_{st.session_state.enroll_segment}"):
                        # Clear the stored audio_data
                        del st.session_state['enroll_audio_data']
                        st.experimental_rerun()
                else:
                    st.write("Click the 'Enter' button to start recording.")
            else:
                # Enrollment completed
                user_dir = f'vault_data/{user_id}'
                os.makedirs(user_dir, exist_ok=True)

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
                st.success(f"User {user_id} enrolled successfully with {len(embeddings)} segments.")
                # Clear enrollment session state
                del st.session_state.enroll_embeddings
                del st.session_state.enroll_segment
                del st.session_state.enroll_user_id

def verify_user():
    """Verify a user by comparing their new recording with saved embeddings."""
    st.write("Verify a user")

    user_id = st.text_input("Enter user ID to verify")

    if user_id:
        if user_id not in user_data:
            st.warning(f"User {user_id} not found!")
            return

        # Record a new sample for verification
        st.write(f"\nRecording verification audio for {user_id}...")

        # Add an "Enter" button to start recording
        if st.button("Enter"):
            audio_data = record_audio_with_timer(duration=5)
            # Store audio_data in session_state
            st.session_state['verification_audio_data'] = audio_data
            # Display audio player
            st.audio(audio_data, format='audio/wav', sample_rate=16000)

        # Check if audio_data is in session_state
        if 'verification_audio_data' in st.session_state:
            audio_data = st.session_state['verification_audio_data']
            # Display audio player
            st.audio(audio_data, format='audio/wav', sample_rate=16000)
            if st.button("Verify"):
                verification_embedding = compute_embedding(audio_data)

                # Compare with each saved embedding and find the highest similarity
                max_similarity = 0
                best_match_idx = -1

                for idx, embedding in enumerate(user_data[user_id]['embeddings']):
                    similarity = compare_embeddings(embedding, verification_embedding)
                    st.write(f"Similarity with embedding {idx + 1}: {similarity:.2f}")

                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match_idx = idx

                st.write(f"\nBest match similarity: {max_similarity:.2f} with embedding {best_match_idx + 1}")

                if max_similarity > SIMILARITY_THRESHOLD_VERIFY:
                    st.success(f"Verification successful for {user_id} with embedding {best_match_idx + 1}.")

                    # Update the matching embedding using Continual Learning
                    old_embedding = user_data[user_id]['embeddings'][best_match_idx]
                    updated_embedding = update_embedding_incrementally(old_embedding, verification_embedding)
                    user_data[user_id]['embeddings'][best_match_idx] = updated_embedding
                    torch.save(
                        updated_embedding,
                        f'vault_data/{user_id}/{user_id}_embedding_{best_match_idx + 1}.pt',
                    )
                    st.write(f"Embedding {best_match_idx + 1} updated with Continual Learning.")

                    # Increment verification count
                    user_data[user_id]['verification_count'] += 1

                    # Save updated verification count
                    user_dir = f'vault_data/{user_id}'
                    with open(f'{user_dir}/verification_count.txt', 'w') as f:
                        f.write(str(user_data[user_id]['verification_count']))

                    # Update memory buffer
                    user_data[user_id]['memory_buffer'].append(
                        (best_match_idx, verification_embedding)
                    )
                    # Limit memory buffer size if necessary
                    MAX_MEMORY_SIZE = 50
                    if len(user_data[user_id]['memory_buffer']) > MAX_MEMORY_SIZE:
                        user_data[user_id]['memory_buffer'].pop(0)  # Remove oldest

                    # Perform Rehearsal-based Learning every 10 successful verifications
                    if user_data[user_id]['verification_count'] % 10 == 0:
                        st.write("Performing Rehearsal-based Learning...")
                        perform_rehearsal_learning(user_id)
                        st.write(
                            f"User {user_id}'s embeddings updated with Rehearsal-based Learning."
                        )
                else:
                    st.error(f"Verification failed.")

                # Clear the stored audio_data after verification
                del st.session_state['verification_audio_data']
        else:
            st.write("Click the 'Enter' button to start recording.")

def remove_user():
    """Remove a user and their vault."""
    st.write("Remove a user")

    user_id = st.text_input("Enter user ID to remove")

    if user_id:
        if user_id in user_data:
            del user_data[user_id]
            user_dir = f'vault_data/{user_id}'
            if os.path.exists(user_dir):
                for file in os.listdir(user_dir):
                    os.remove(os.path.join(user_dir, file))
                os.rmdir(user_dir)
            st.success(f"User {user_id} and their data have been removed.")
        else:
            st.warning(f"User {user_id} not found.")

def main():
    st.title("Voice Authentication System")
    load_existing_users()  # Load existing users on start

    menu_options = ["Enroll a new user", "Verify a user", "Remove a user"]
    choice = st.sidebar.selectbox("Menu", menu_options)

    if choice == "Enroll a new user":
        enroll_user()
    elif choice == "Verify a user":
        verify_user()
    elif choice == "Remove a user":
        remove_user()
    else:
        st.write("Select an option from the menu.")

if __name__ == "__main__":
    main()


# import streamlit as st
# import sounddevice as sd
# from scipy.io.wavfile import write, read
# from speechbrain.pretrained import SpeakerRecognition
# import torch
# import os
# import numpy as np
# from io import BytesIO
# from collections import defaultdict
# import warnings

# warnings.filterwarnings("ignore")

# # Initialize SpeakerRecognition model
# verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")

# # User data storage
# user_data = defaultdict(list)

# # Thresholds
# SIMILARITY_THRESHOLD_ENROLL = 0.50
# SIMILARITY_THRESHOLD_VERIFY = 0.70

# # Streamlit page setup
# st.title("Voice Authentication System")

# def record_audio(duration=5, fs=16000):
#     """Record audio using the microphone for a specified duration."""
#     audio_data = np.empty((int(duration * fs), 1), dtype=np.float32)
#     stream = sd.InputStream(samplerate=fs, channels=1)
    
#     with stream:
#         st.write("Recording... Please wait")
#         audio_data = stream.read(int(duration * fs))[0]

#     st.success("Recording completed!")
#     return audio_data

# def compute_embedding(audio_data):
#     """Helper function to compute the embedding of the given audio."""
#     byte_io = BytesIO()
#     write(byte_io, 16000, audio_data)
#     byte_io.seek(0)

#     audio_array = np.array(read(byte_io)[1])
#     if audio_array.ndim > 1:
#         audio_array = audio_array[:, 0]

#     audio_tensor = torch.tensor(audio_array).unsqueeze(0)
#     embedding = verification.encode_batch(audio_tensor)
#     return embedding

# def compare_embeddings(embedding1, embedding2):
#     """Compare two embeddings and return the similarity score."""
#     similarity = verification.similarity(embedding1, embedding2)
#     return similarity.item()

# def enroll_user(user_id):
#     """Enroll a user by recording multiple segments."""
#     if user_id in user_data:
#         st.error(f"User {user_id} already exists!")
#         return

#     user_dir = f'vault_data/{user_id}'
#     os.makedirs(user_dir, exist_ok=True)

#     embeddings = []

#     for i in range(5):
#         st.write(f"\nRecording segment {i + 1} for {user_id}...")
#         segment = record_audio(duration=5)

#         # Ask if the user wants to re-record this segment
#         re_record = st.radio(f"Do you want to re-record segment {i + 1}?", options=["No", "Yes"])

#         while re_record == "Yes":
#             segment = record_audio(duration=5)
#             re_record = st.radio(f"Do you want to re-record segment {i + 1}?", options=["No", "Yes"])

#         embedding = compute_embedding(segment)
#         embeddings.append(embedding)

#         if i > 0:
#             similarity = compare_embeddings(embeddings[i - 1], embeddings[i])
#             st.write(f"Similarity with previous segment: {similarity:.2f}")
#             if similarity < SIMILARITY_THRESHOLD_ENROLL:
#                 st.warning(f"Low similarity ({similarity:.2f}) detected. You may want to re-record.")
#                 re_record = st.radio(f"Do you want to re-record segment {i + 1} due to low similarity?", options=["No", "Yes"])
#                 while re_record == "Yes":
#                     segment = record_audio(duration=5)
#                     embedding = compute_embedding(segment)
#                     embeddings[i] = embedding
#                     similarity = compare_embeddings(embeddings[i - 1], embeddings[i])
#                     st.write(f"Similarity with previous segment: {similarity:.2f}")
#                     re_record = st.radio(f"Do you want to re-record segment {i + 1}?", options=["No", "Yes"])

#     for idx, embedding in enumerate(embeddings):
#         torch.save(embedding, f'{user_dir}/{user_id}_embedding_{idx + 1}.pt')

#     user_data[user_id] = embeddings
#     st.success(f"User {user_id} enrolled with {len(embeddings)} segments!")

# def verify_user(user_id):
#     """Verify a user by comparing their new recording with saved embeddings."""
#     if user_id not in user_data:
#         st.error(f"User {user_id} not found!")
#         return

#     st.write(f"Recording verification audio for {user_id}...")
#     verification_segment = record_audio(duration=5)
#     verification_embedding = compute_embedding(verification_segment)

#     max_similarity = 0
#     best_match_idx = -1

#     for idx, embedding in enumerate(user_data[user_id]):
#         similarity = compare_embeddings(embedding, verification_embedding)
#         st.write(f"Similarity with embedding {idx + 1}: {similarity:.2f}")

#         if similarity > max_similarity:
#             max_similarity = similarity
#             best_match_idx = idx

#     st.write(f"Best match similarity: {max_similarity:.2f} with embedding {best_match_idx + 1}")

#     if max_similarity > SIMILARITY_THRESHOLD_VERIFY:
#         st.success(f"Verification successful for {user_id} with embedding {best_match_idx + 1}.")
#         user_data[user_id][best_match_idx] = verification_embedding
#         torch.save(verification_embedding, f'vault_data/{user_id}/{user_id}_embedding_{best_match_idx + 1}.pt')
#         st.success(f"Embedding {best_match_idx + 1} updated.")
#     else:
#         st.error(f"Verification failed. Similarity did not meet the threshold ({SIMILARITY_THRESHOLD_VERIFY}).")

# def load_existing_users():
#     """Load existing users from the vault_data directory."""
#     if not os.path.exists('vault_data'):
#         os.makedirs('vault_data')

#     for user_id in os.listdir('vault_data'):
#         user_dir = f'vault_data/{user_id}'
#         if os.path.isdir(user_dir):
#             embeddings = []
#             for file in os.listdir(user_dir):
#                 if file.endswith(".pt"):
#                     embeddings.append(torch.load(os.path.join(user_dir, file)))
#             if embeddings:
#                 user_data[user_id] = embeddings
#                 st.success(f"User {user_id} loaded with {len(embeddings)} embeddings.")

# def remove_user(user_id):
#     """Remove a user and their data."""
#     if user_id in user_data:
#         del user_data[user_id]
#         user_dir = f'vault_data/{user_id}'
#         if os.path.exists(user_dir):
#             for file in os.listdir(user_dir):
#                 os.remove(os.path.join(user_dir, file))
#             os.rmdir(user_dir)
#         st.success(f"User {user_id} and their data have been removed.")

# # Load existing users automatically on app startup
# load_existing_users()

# # Streamlit menu
# option = st.selectbox(
#     "Select an option", 
#     ("Enroll a new user", "Verify a user", "Remove a user")
# )

# if option == "Enroll a new user":
#     user_id = st.text_input("Enter user ID to enroll")
#     if st.button("Enroll"):
#         enroll_user(user_id)

# elif option == "Verify a user":
#     user_id = st.text_input("Enter user ID to verify")
#     if st.button("Verify"):
#         verify_user(user_id)

# elif option == "Remove a user":
#     user_id = st.text_input("Enter user ID to remove")
#     if st.button("Remove"):
#         remove_user(user_id)
