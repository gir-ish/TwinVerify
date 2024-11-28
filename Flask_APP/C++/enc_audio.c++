#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdlib>   // For rand()
#include <ctime>     // For time()
#include <chrono>    // For measuring time
#include <unordered_map>  // For memoization

using namespace std;
using namespace std::chrono;

// Function to compute the encrypted value based on a byte value
int encryptByte(int byteValue) {
    int ones = byteValue % 10;
    int tens = (byteValue / 10) % 10;
    int hundreds = (byteValue / 100) % 10;

    int result = (ones * ones) + (tens * tens * tens) + (hundreds * hundreds);
    return result;
}

// Function to calculate the two-digit differentiator for encryption
string calculateDifferentiator(int byteValue) {
    int ones = byteValue % 10;
    int tens = (byteValue / 10) % 10;
    int hundreds = (byteValue / 100) % 10;
    int differentiator = (hundreds * 10) + ones;

    // Ensure it's always two digits by adding a leading zero if needed
    stringstream ss;
    ss << (differentiator < 10 ? "0" : "") << differentiator;

    return ss.str();
}

// Function to handle the encryption of audio data with memoization
string encryptAndMapToChar(const vector<char>& audioData) {
    stringstream encryptedMessage; // Use stringstream to build the result
    srand(time(0)); // Seed for randomness

    // Memoization map to store previously encrypted characters
    unordered_map<int, string> memo;

    for (unsigned char byte : audioData) {
        int byteValue = static_cast<int>(byte);

        if (memo.find(byteValue) != memo.end()) {
            // If byte is already encrypted, use the stored result
            encryptedMessage << memo[byteValue];
        } else {
            // Encrypt the byte and store the result in the memo
            int encryptedValue = encryptByte(byteValue);
            string differentiator = calculateDifferentiator(byteValue);

            // Convert encrypted value to string and pad with leading zeros to ensure 3 digits
            stringstream encryptedValueStr;
            encryptedValueStr << (encryptedValue < 100 ? (encryptedValue < 10 ? "00" : "0") : "") << encryptedValue;

            // Append the final 5-digit encrypted result
            string result = encryptedValueStr.str() + differentiator;
            memo[byteValue] = result;  // Store the result in memo
            encryptedMessage << result;
        }
    }

    return encryptedMessage.str(); // Return the result as a string
}

// Function to write both metadata and encrypted data to a text file
void writeEncryptedToFile(const string& filename, const string& metadata, const string& encryptedData) {
    ofstream file(filename);
    if (!file) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1); // Exit if file can't be opened
    }

    // Write metadata and encrypted data to the file
    file << metadata << "," << encryptedData;
    file.close();
}

// Function to convert a WAV file to a text format and encrypt the data
void convertAndEncryptWAV(const string& inputFilename, const string& outputFilename) {
    ifstream inputFile(inputFilename, ios::binary);

    if (!inputFile) {
        cerr << "Could not open the file: " << inputFilename << endl;
        exit(1);
    }

    // Read the WAV header (44 bytes)
    const int headerSize = 44;
    vector<char> headerData(headerSize);
    inputFile.read(headerData.data(), headerSize);

    // Extract important metadata from the header
    int sampleRate = *reinterpret_cast<int*>(&headerData[24]);
    short numChannels = *reinterpret_cast<short*>(&headerData[22]);
    short bitsPerSample = *reinterpret_cast<short*>(&headerData[34]);

    // Move the file pointer to the end to determine file size
    inputFile.seekg(0, ios::end);
    size_t fileSize = inputFile.tellg();
    inputFile.seekg(headerSize, ios::beg);

    // Calculate the size of the raw audio data
    size_t dataSize = fileSize - headerSize;

    // Create a vector to hold the audio data
    vector<char> audioData(dataSize);

    // Read the data into the vector
    inputFile.read(audioData.data(), dataSize);

    // Close the WAV file
    inputFile.close();

    // Store metadata in a string
    stringstream metadataStream;
    metadataStream << sampleRate << " " << numChannels << " " << bitsPerSample;
    string metadata = metadataStream.str();

    // Encrypt the audio data
    string encryptedData = encryptAndMapToChar(audioData);

    // Write the metadata and encrypted audio data to the output file
    writeEncryptedToFile(outputFilename, metadata, encryptedData);

    cout << "Encryption completed and saved to: " << outputFilename << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_wav_filename>" << endl;
        return 1;
    }

    string inputFilename = argv[1];
    string outputFilename = "encrypted_audio_data.txt";

    // Start measuring time
    auto start = high_resolution_clock::now();

    // Convert and encrypt the WAV file
    convertAndEncryptWAV(inputFilename, outputFilename);

    // Stop measuring time
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Encryption completed in " << duration.count() << " milliseconds" << endl;

    return 0;
}
