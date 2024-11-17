import sounddevice as sd
from scipy.io.wavfile import write, read
from speechbrain.inference import SpeakerRecognition
import torch
import os
import numpy as np
from io import BytesIO
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings

warnings.filterwarnings("ignore")

# Load the speaker verification model
verification = SpeakerRecognition.from_hparams(source="microsoft/unispeech-sat-base-100h-libri-ft", savedir="tmp")

# Store enrolled users and their data
user_data = {}

# Custom environment for PPO
class SpeakerVerificationEnv(gym.Env):
    """Custom Environment that follows gym interface for PPO."""
    def __init__(self, user_id):
        super(SpeakerVerificationEnv, self).__init__()
        self.user_id = user_id
        self.action_space = gym.spaces.Discrete(3)  # 0: decrease threshold, 1: keep, 2: increase threshold
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Similarity score
        self.thresholds = np.linspace(0.5, 0.9, 10)  # Threshold range
        self.state = 5  # Initial state (middle threshold)
        self.threshold = self.thresholds[self.state]
        self.done = False

    def reset(self):
        """Reset the environment state."""
        self.state = 5  # Start from the middle threshold
        self.threshold = self.thresholds[self.state]
        return np.array([self.threshold], dtype=np.float32)

    def step(self, action):
        """Apply the action (adjust threshold) and return the new state, reward, and done status."""
        if action == 0:  # Decrease threshold
            self.state = max(0, self.state - 1)
        elif action == 2:  # Increase threshold
            self.state = min(len(self.thresholds) - 1, self.state + 1)

        self.threshold = self.thresholds[self.state]

        # Compute the similarity score using the verification model
        similarity_score = self.get_similarity_score()

        # Decision logic
        reward, self.done = self.compute_reward(similarity_score)
        
        return np.array([self.threshold], dtype=np.float32), reward, self.done, {}

    def compute_reward(self, similarity_score):
        """Compute the reward based on the similarity score and the current threshold."""
        if similarity_score >= self.threshold:
            return 1, True  # Correct verification (true positive)
        else:
            return -5, True  # False positive (critical error) or false negative

    def get_similarity_score(self):
        """Calculate the similarity score between stored embedding and new input."""
        user_embedding = torch.load(f'vault_data/{self.user_id}/{self.user_id}_embedding.pt')
        similarity = verification.similarity(self.new_embedding, user_embedding)
        return similarity.item()

    def set_new_embedding(self, embedding):
        """Set the new embedding for verification."""
        self.new_embedding = embedding


# Record audio and return as NumPy array
def record_audio_to_memory(duration=5, fs=16000):
    """Record audio and return it as a NumPy array."""
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording completed.")
    return audio_data

# Compute embedding from audio
def compute_embedding(audio_data):
    """Helper function to compute the embedding of the given audio."""
    byte_io = BytesIO()
    write(byte_io, 16000, audio_data)  # Store audio in memory (not on disk)
    byte_io.seek(0)  # Move pointer to the start
    embedding = verification.encode_batch(torch.tensor(np.array(read(byte_io)[1])).unsqueeze(0))
    return embedding

# Compare two embeddings and return similarity score
def compare_embeddings(embedding1, embedding2):
    """Compare two embeddings and return the similarity score."""
    similarity = verification.similarity(embedding1, embedding2)
    return similarity.item()

# Enroll user by recording voice 5 times and training PPO during enrollment
def enroll_user(user_id):
    """Enroll a new user by recording their voice 5 times, ensuring consistency and training PPO during enrollment."""
    if user_id in user_data:
        print(f"User {user_id} already exists!")
        return

    user_dir = f'vault_data/{user_id}'
    os.makedirs(user_dir, exist_ok=True)

    embeddings = []
    avg_similarity_threshold = 0.86  
    for i in range(5):
        while True:  # Loop to give the user another chance if similarity is low
            print(f"Recording {i+1} of 5 for {user_id}...")
            audio_data = record_audio_to_memory(duration=5)
            embedding = compute_embedding(audio_data)

            # If it's the first recording, we can't compare it to anything
            if i == 0:
                embeddings.append(embedding)
                print("First recording saved.")
                break  # Move to the next recording

            # Compare with the average of previous embeddings
            avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
            similarity = compare_embeddings(embedding, avg_embedding)

            print(f"Similarity with average of previous recordings: {similarity:.2f}")
            
            if similarity >= avg_similarity_threshold:
                embeddings.append(embedding)
                print("Recording is consistent and saved.")
                break
            else:
                print(f"Similarity {similarity:.2f} is below the threshold {avg_similarity_threshold}. Please try again.")

    # Average embeddings to create a final embedding
    final_embedding = torch.mean(torch.stack(embeddings), dim=0)
    torch.save(final_embedding, f'{user_dir}/{user_id}_embedding.pt')

    user_data[user_id] = f'Vault for {user_id}'

    # Initialize the environment and PPO model
    env = SpeakerVerificationEnv(user_id)
    env = DummyVecEnv([lambda: env])
    model_path = f'vault_data/{user_id}/{user_id}_ppo_model'

    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path)
        print(f"PPO model loaded for user {user_id}.")
    else:
        model = PPO('MlpPolicy', env, verbose=1)
        print(f"New PPO model initialized for user {user_id}.")

    # Train PPO using enrollment embeddings
    print("Training PPO model during enrollment...")
    for i in range(5):
        env.set_new_embedding(embeddings[i])
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, _ = env.step(action)
        model.learn(total_timesteps=500)  # Learning from enrollment data

    model.save(model_path)
    print(f"PPO model trained and saved for user {user_id} during enrollment.")
    print(f"User {user_id} enrolled successfully with averaged embeddings.")

# Verify and open the vault
def verify_vault(user_id):
    """Verify a user by asking for a voice input and comparing it with the stored embedding."""
    if user_id not in user_data:
        print(f"User {user_id} not found.")
        return

    print(f"Recording for verification...")
    audio_data = record_audio_to_memory(duration=5)
    new_embedding = compute_embedding(audio_data)

    # Load user's embedding and initialize PPO
    user_dir = f'vault_data/{user_id}'
    if not os.path.exists(f'{user_dir}/{user_id}_embedding.pt'):
        print(f"User {user_id} has no enrolled data.")
        return

    env = SpeakerVerificationEnv(user_id)
    env.set_new_embedding(new_embedding)
    env = DummyVecEnv([lambda: env])
    model_path = f'{user_dir}/{user_id}_ppo_model'

    if not os.path.exists(model_path + ".zip"):
        print(f"No trained PPO model for {user_id}.")
        return

    model = PPO.load(model_path)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)

    if reward == 1:
        print(f"Access granted to {user_id}'s vault.")
        open_vault(user_id)
    else:
        print(f"Access denied for {user_id}.")

# Open the user's vault
def open_vault(user_id):
    """Open the user's vault."""
    print(f"{user_data[user_id]} opened!")

# Remove a user
def remove_user(user_id):
    """Remove a user and delete their data from the system."""
    if user_id in user_data:
        del user_data[user_id]
        user_dir = f'vault_data/{user_id}'
        if os.path.exists(user_dir):
            for file in os.listdir(user_dir):
                os.remove(os.path.join(user_dir, file))
            os.rmdir(user_dir)
        print(f"User {user_id} removed successfully.")
    else:
        print(f"User {user_id} not found.")

# Load existing users at startup
def load_existing_users():
    """Load existing users and their embeddings from disk at startup."""
    if not os.path.exists('vault_data'):
        os.makedirs('vault_data')

    for user_id in os.listdir('vault_data'):
        user_dir = f'vault_data/{user_id}'
        if os.path.isdir(user_dir):
            embedding_path = f'{user_dir}/{user_id}_embedding.pt'
            if os.path.exists(embedding_path):
                user_data[user_id] = f'Vault for {user_id}'
                print(f"User {user_id} loaded successfully.")

# Menu for system interaction
def menu():
    """Main menu to interact with the system."""
    load_existing_users()  # Load all existing users at startup
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
