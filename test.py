import sounddevice as sd
from scipy.io.wavfile import write
from speechbrain.pretrained import SpeakerRecognition
import torch
enrolled_embedding = torch.load("vault_data/user1/user1_embedding.pt")
print(f"Shape of the enrolled embedding: {enrolled_embedding.shape}")