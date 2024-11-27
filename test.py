# import sounddevice as sd
# from scipy.io.wavfile import write
# from speechbrain.pretrained import SpeakerRecognition
# import torch
# enrolled_embedding = torch.load("vault_data/user1/user1_embedding.pt")
# print(f"Shape of the enrolled embedding: {enrolled_embedding.shape}")

import os

def display_folder_structure(root_folder, indent=""):
    try:
        items = os.listdir(root_folder)
    except PermissionError:
        print(f"{indent}[Access Denied]")
        return

    for index, item in enumerate(items):
        path = os.path.join(root_folder, item)
        if os.path.isdir(path):
            print(f"{indent}📁 {item}")
            display_folder_structure(path, indent + "    ")
        else:
            print(f"{indent}📄 {item}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path to display its structure: ")
    if os.path.exists(folder_path):
        print(f"Folder structure of: {folder_path}")
        display_folder_structure(folder_path)
    else:
        print("The provided folder path does not exist.")
