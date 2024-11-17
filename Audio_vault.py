import sounddevice as sd
from scipy.io.wavfile import write, read
from speechbrain.pretrained import SpeakerRecognition
import torch
import os
import numpy as np
from io import BytesIO
import random
import warnings
warnings.filterwarnings("ignore")

verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp")


user_data = {}
user_rl_agents = {} 

class VerificationRLAgent:
    def __init__(self, actions, state_size, user_id):
        self.actions = actions  # Possible actions: increase or decrease threshold
        self.state_size = state_size  # Number of states (threshold levels)
        self.q_table = np.zeros((state_size, len(actions)))  # Initialize Q-table
        self.learning_rate = 0.1  # Learning rate for Q-learning
        self.discount = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.epsilon_min = 0.01  # Minimum value for epsilon
        self.user_id = user_id  # User ID associated with the RL agent
        self.load_q_table()  # Load previous Q-table if it exists

    def choose_action(self, state):
        # Epsilon-greedy strategy for exploration and exploitation
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)  # Explore: Choose random action
        q_values = self.q_table[state]
        return np.argmax(q_values)  # Exploit: Choose the best action

    def learn(self, state, action, reward, next_state):
        # Update Q-value based on reward received
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        
        # Reduce exploration rate (epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Save Q-table after every learning step
        self.save_q_table()

    def save_q_table(self):
        """Save the Q-table to a file for persistence."""
        file_path = f'vault_data/{self.user_id}/{self.user_id}_qtable.npy'
        np.save(file_path, self.q_table)
        print(f"Q-table for user {self.user_id} saved to {file_path}")

    def load_q_table(self):
        """Load the Q-table from a file if it exists."""
        file_path = f'vault_data/{self.user_id}/{self.user_id}_qtable.npy'
        if os.path.exists(file_path):
            self.q_table = np.load(file_path)
            print(f"Q-table for user {self.user_id} loaded from {file_path}")
        else:
            print(f"No Q-table found for user {self.user_id}. Starting fresh.")

def record_audio_to_memory(duration=5, fs=16000):
    """Record audio and return it as a NumPy array."""
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print(f"Recording completed.")
    return audio_data

def save_wav_in_vault(user_dir, user_id, audio_data, fs=16000):
    """Save the recorded audio data as a .wav file inside the user's vault."""
    file_path = f'{user_dir}/{user_id}_enrollment.wav'
    write(file_path, fs, audio_data)  # Save the audio data as a wav file
    print(f"Enrollment saved to {file_path}")
    return file_path

def compare_embeddings(embedding1, embedding2):
    """Compare two embeddings and return the similarity score."""
    similarity = verification.similarity(embedding1, embedding2)
    return similarity.item()  # Convert tensor to Python float


def compute_embedding(audio_data):
    """Helper function to compute the embedding of the given audio."""
    byte_io = BytesIO()
    write(byte_io, 16000, audio_data)  # Store audio in memory (not on disk)
    byte_io.seek(0)  # Move pointer to the start
    embedding = verification.encode_batch(torch.tensor(np.array(read(byte_io)[1])).unsqueeze(0))
    return embedding

def enroll_user(user_id):
    """Enroll a new user by recording their voice 5 times with a retry option if threshold isn't met."""
    if user_id in user_data:
        print(f"User {user_id} already exists!")
        return

    user_dir = f'vault_data/{user_id}'
    os.makedirs(user_dir, exist_ok=True)

    embeddings = []
    threshold = 0.75  # Similarity threshold for consistency between recordings

    for i in range(5):
        while True:  # Loop until the user records a consistent input
            print(f"Recording {i+1} of 5 for {user_id}...")
            audio_data = record_audio_to_memory(duration=5)
            embedding = compute_embedding(audio_data)

            # If it's the first recording, there's nothing to compare
            if i == 0:
                embeddings.append(embedding)
                print("First recording saved.")
                break  # Move to the next recording

            # Compare with the previous recording
            similarity = compare_embeddings(embeddings[-1], embedding)
            print(f"Similarity between previous recording and current: {similarity:.2f}")

            if similarity >= threshold:
                print(f"Recordings are consistent. Saving this recording.")
                embeddings.append(embedding)
                break  # Proceed to the next recording
            else:
                print(f"Inconsistency detected: Similarity {similarity:.2f} is below threshold {threshold}. Please record again.")
    
    # Average embeddings to create a final embedding
    final_embedding = torch.mean(torch.stack(embeddings), dim=0)

    # Save the final embedding for the user
    torch.save(final_embedding, f'{user_dir}/{user_id}_embedding.pt')

    # Initialize RL agent for this user
    user_data[user_id] = f'Vault for {user_id}'
    user_rl_agents[user_id] = VerificationRLAgent(actions=[-1, 0, 1], state_size=10, user_id=user_id)

    print(f"User {user_id} enrolled successfully with averaged embeddings.")

# def enroll_user(user_id):
#     """Enroll a new user by recording their voice 5 times, ensuring consistency."""
#     if user_id in user_data:
#         print(f"User {user_id} already exists!")
#         return

#     user_dir = f'vault_data/{user_id}'
#     os.makedirs(user_dir, exist_ok=True)

#     # Record 5 audio samples and compute embeddings
#     embeddings = []
#     for i in range(5):
#         print(f"Recording {i+1} of 5 for {user_id}...")
#         audio_data = record_audio_to_memory(duration=5)
#         embedding = compute_embedding(audio_data)
#         embeddings.append(embedding)

#     # Compare embeddings for consistency and print similarity scores
#     consistent = True
#     threshold = 0.86  
#     print("\nChecking consistency between recordings:")

#     for i in range(4):
#         similarity = compare_embeddings(embeddings[i], embeddings[i+1])
#         print(f"Similarity between recording {i+1} and {i+2}: {similarity:.2f}")
        
#         if similarity < threshold:
#             consistent = False
#             print(f"Inconsistency detected: Similarity {similarity:.2f} is below threshold {threshold}.")
#         else:
#             print(f"Recordings {i+1} and {i+2} are consistent.")

#     if not consistent:
#         print(f"Inconsistent recordings for {user_id}. Please re-enroll.")
#         return

#     # Average embeddings to create a final embedding
#     final_embedding = torch.mean(torch.stack(embeddings), dim=0)

#     # Save the final embedding for the user
#     torch.save(final_embedding, f'{user_dir}/{user_id}_embedding.pt')

#     # Initialize RL agent for this user
#     user_data[user_id] = f'Vault for {user_id}'
#     user_rl_agents[user_id] = VerificationRLAgent(actions=[-1, 0, 1], state_size=10, user_id=user_id)

#     print(f"User {user_id} enrolled successfully with averaged embeddings.")

def load_existing_users():
    """Load existing users and their RL agents from the vault directory on application start."""
    if not os.path.exists('vault_data'):
        os.makedirs('vault_data')

    # Check each subdirectory in the vault_data folder
    for user_id in os.listdir('vault_data'):
        user_dir = f'vault_data/{user_id}'
        if os.path.isdir(user_dir):
            # Load the embeddings for each user
            embedding_path = f'{user_dir}/{user_id}_embedding.pt'
            if os.path.exists(embedding_path):
                user_data[user_id] = f'Vault for {user_id}'
                # Initialize RL agent for this user (loads Q-table if it exists)
                user_rl_agents[user_id] = VerificationRLAgent(actions=[-1, 0, 1], state_size=10, user_id=user_id)
                print(f"User {user_id} loaded successfully.")

def remove_user(user_id):
    """Remove a user and their vault."""
    if user_id in user_data:
        del user_data[user_id]
        del user_rl_agents[user_id]
        user_dir = f'vault_data/{user_id}'
        if os.path.exists(user_dir):
            for file in os.listdir(user_dir):
                os.remove(os.path.join(user_dir, file))
            os.rmdir(user_dir)
        print(f"User {user_id} removed successfully.")
    else:
        print(f"User {user_id} not found.")

def open_vault(user_id):
    """Open the user's vault."""
    print(f"{user_data[user_id]} opened!")

def verify_vault(user_id):
    """Verify a user by asking for a voice input and comparing it with the stored embedding."""
    if user_id not in user_data:
        print(f"User {user_id} not found.")
        return

    # Record verification audio in memory
    print(f"Recording for verification...")
    audio_data = record_audio_to_memory(duration=5)
    
    # Convert the recorded data to an in-memory WAV file using BytesIO
    byte_io = BytesIO()
    write(byte_io, 16000, audio_data)  # Store audio in memory (not on disk)
    byte_io.seek(0)  # Move pointer to the start

    # Load the in-memory audio for verification
    new_embedding = verification.encode_batch(torch.tensor(np.array(read(byte_io)[1])).unsqueeze(0))

    # Load stored embedding of the specified user
    user_embedding = torch.load(f'vault_data/{user_id}/{user_id}_embedding.pt')

    # RL Agent for this user
    rl_agent = user_rl_agents[user_id]
    
    # Set threshold states (0-9) and thresholds corresponding to states
    thresholds = np.linspace(0.5, 0.9, 10)
    state = 5  # Initial state (middle threshold)
    threshold = thresholds[state]
    
    # Compute similarity score
    score = verification.similarity(new_embedding, user_embedding)

    # Choose action: decrease, keep, or increase threshold
    action = rl_agent.choose_action(state)
    next_state = state + action
    next_state = np.clip(next_state, 0, len(thresholds) - 1)  # Ensure valid state
    threshold = thresholds[next_state]

    # Decision based on similarity score and threshold
    correct_user = score >= threshold  # Did the system verify this user as legitimate?
    
    impostor_attempt = user_id not in user_data  # Is this user a legitimate one?

    if correct_user and not impostor_attempt:
        # True Positive: Legitimate user verified correctly
        reward = 1  # Positive reward for correct verification
        print(f"Access granted to {user_data[user_id]} (Score: {score})")
        open_vault(user_id)
    elif not correct_user and not impostor_attempt:
        # False Negative: Legitimate user denied access
        reward = -1  # Negative reward for incorrect denial
        print(f"Access denied to legitimate user {user_data[user_id]} (Score: {score})")
    elif correct_user and impostor_attempt:
        # False Positive: Impostor granted access (serious error)
        reward = -2  # Large negative reward for serious security breach
        print(f"Access granted to impostor! Critical Error! (Score: {score})")
    else:
        # True Negative: Impostor correctly denied access
        reward = 1  # Positive reward for denying impostor
        print(f"Access denied to impostor (Score: {score})")

    # RL Agent learns from the result
    rl_agent.learn(state, action, reward, next_state)


def menu():
    """Main menu to interact with the system."""
    load_existing_users()  # Load all existing users and their RL agents on startup
    while True:
        print("\n==== Voice Vault System ====")
        print("1. Enroll User")
        print("2. Verify and Open Vault")
        print("3. Remove User")
        print("4. Exit")
        
        choice = input("Please select an option (1/2/3/4): ")

        if choice == "1":
            user_id = input("Enter a unique User ID: ")
            enroll_user(user_id)
        elif choice == "2":
            user_id = input("Enter User ID for verification: ")
            verify_vault(user_id)
        elif choice == "3":
            user_id = input("Enter User ID to remove: ")
            remove_user(user_id)
        elif choice == "4":
            print("Exiting the system.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    menu()
