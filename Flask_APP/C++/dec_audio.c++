#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <unordered_map>  // For memoization

using namespace std;

// Function to decrypt a 5-digit code using memoization
char decryptCode(const string& code, unordered_map<string, char>& memo) {
    // Check if the code has been decrypted before (memoization)
    if (memo.find(code) != memo.end()) {
        return memo[code];
    }

    int num = stoi(code);

    if (num >= 90000 && num <= 99999) {
        // Handle blank space
        memo[code] = ' ';  // Memoize the result
        return ' ';
    } else {
        // Extract parts from the code
        int encryptedValue = stoi(code.substr(0, 3));
        int hundreds = stoi(code.substr(3, 1));
        int ones = stoi(code.substr(4, 1));

        // Calculate the tens digit
        int sumOfSquares = (hundreds * hundreds) + (ones * ones);
        int difference = encryptedValue - sumOfSquares;

        // Find the cube root of the difference to get the tens digit
        int tens = round(cbrt(difference));

        // Reconstruct the byte value (ASCII value)
        int byteValue = (hundreds * 100) + (tens * 10) + ones;

        // Store the decrypted byte in the memoization map
        char decryptedChar = static_cast<char>(byteValue);
        memo[code] = decryptedChar;  // Memoize the result

        return decryptedChar;
    }
}

// Function to decrypt the audio data part of the encrypted message using memoization
vector<char> decryptAudioData(const string& encryptedMessage) {
    vector<char> decryptedData;
    size_t pos = 0;
    size_t nextPos;

    // Memoization map to store previously decrypted codes
    unordered_map<string, char> memo;

    while (pos < encryptedMessage.length()) {
        nextPos = pos + 5;
        if (nextPos > encryptedMessage.length()) {
            // Handle unexpected end of string
            break;
        }

        string code = encryptedMessage.substr(pos, 5);
        decryptedData.push_back(decryptCode(code, memo));  // Pass the memo map
        pos = nextPos;
    }

    return decryptedData;
}

// Function to write a WAV header
void writeWAVHeader(ofstream &outputFile, int dataSize, int sampleRate, int numChannels, int bitsPerSample) {
    int byteRate = sampleRate * numChannels * bitsPerSample / 8;
    int blockAlign = numChannels * bitsPerSample / 8;
    int chunkSize = 36 + dataSize; // 36 + data size (since the header is always 44 bytes)
    int subchunk2Size = dataSize;

    // RIFF chunk descriptor
    outputFile.write("RIFF", 4);                      // ChunkID "RIFF"
    outputFile.write(reinterpret_cast<const char *>(&chunkSize), 4);   // ChunkSize
    outputFile.write("WAVE", 4);                      // Format "WAVE"

    // fmt sub-chunk
    outputFile.write("fmt ", 4);                      // Subchunk1ID "fmt "
    int subchunk1Size = 16;                           // Subchunk1Size (PCM)
    outputFile.write(reinterpret_cast<const char *>(&subchunk1Size), 4);
    int16_t audioFormat = 1;                          // AudioFormat (1 for PCM)
    outputFile.write(reinterpret_cast<const char *>(&audioFormat), 2);
    outputFile.write(reinterpret_cast<const char *>(&numChannels), 2); // NumChannels
    outputFile.write(reinterpret_cast<const char *>(&sampleRate), 4);  // SampleRate
    outputFile.write(reinterpret_cast<const char *>(&byteRate), 4);    // ByteRate
    outputFile.write(reinterpret_cast<const char *>(&blockAlign), 2);  // BlockAlign
    outputFile.write(reinterpret_cast<const char *>(&bitsPerSample), 2); // BitsPerSample

    // data sub-chunk
    outputFile.write("data", 4);                      // Subchunk2ID "data"
    outputFile.write(reinterpret_cast<const char *>(&subchunk2Size), 4); // Subchunk2Size
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <encrypted_audio_txt_file>" << endl;
        return 1;
    }

    string inputFilename = argv[1];
    string wavOutputFilename = "reconstructed_audio.wav";

    // Open the encrypted text file containing the audio data and metadata
    ifstream inputFile(inputFilename);
    if (!inputFile) {
        cerr << "Could not open the encrypted text file!" << endl;
        return 1;
    }

    // Read the metadata (sample rate, channels, bit depth)
    int sampleRate, numChannels, bitsPerSample;
    inputFile >> sampleRate >> numChannels >> bitsPerSample;

    // Read the remaining encrypted audio data
    string encryptedMessage;
    getline(inputFile, encryptedMessage, ',');  // Read the metadata line
    getline(inputFile, encryptedMessage);       // Read the encrypted audio data

    // Close the input file
    inputFile.close();

    // Decrypt the audio data
    vector<char> decryptedAudioData = decryptAudioData(encryptedMessage);

    // Calculate the size of the raw audio data
    int dataSize = decryptedAudioData.size();

    // Open a new WAV file to write the audio data back
    ofstream outputFile(wavOutputFilename, ios::binary);
    if (!outputFile) {
        cerr << "Could not create the output WAV file!" << endl;
        return 1;
    }

    // Write the WAV header using the metadata
    writeWAVHeader(outputFile, dataSize, sampleRate, numChannels, bitsPerSample);

    // Write the decrypted audio data back to the new file
    outputFile.write(decryptedAudioData.data(), dataSize);

    // Close the output file
    outputFile.close();

    cout << "Audio data decrypted and written to " << wavOutputFilename << " successfully!" << endl;

    return 0;
}


// #include <iostream>
// #include <fstream>
// #include <sstream>
// #include <vector>
// #include <cmath>
// #include <cstdlib>
// #include <cstdint>

// using namespace std;

// // Function to decrypt a 5-digit code
// char decryptCode(const string& code) {
//     int num = stoi(code);

//     if (num >= 90000 && num <= 99999) {
//         // Handle blank space
//         return ' ';
//     } else {
//         // Extract parts from the code
//         int encryptedValue = stoi(code.substr(0, 3));
//         int hundreds = stoi(code.substr(3, 1));
//         int ones = stoi(code.substr(4, 1));

//         // Calculate the tens digit
//         int sumOfSquares = (hundreds * hundreds) + (ones * ones);
//         int difference = encryptedValue - sumOfSquares;

//         // Find the cube root of the difference to get the tens digit
//         int tens = round(cbrt(difference));

//         // Reconstruct the byte value (ASCII value)
//         int byteValue = (hundreds * 100) + (tens * 10) + ones;

//         // Return the corresponding byte as a char
//         return static_cast<char>(byteValue);
//     }
// }

// // Function to decrypt the audio data part of the encrypted message
// vector<char> decryptAudioData(const string& encryptedMessage) {
//     vector<char> decryptedData;
//     size_t pos = 0;
//     size_t nextPos;

//     while (pos < encryptedMessage.length()) {
//         nextPos = pos + 5;
//         if (nextPos > encryptedMessage.length()) {
//             // Handle unexpected end of string
//             break;
//         }

//         string code = encryptedMessage.substr(pos, 5);
//         decryptedData.push_back(decryptCode(code));
//         pos = nextPos;
//     }

//     return decryptedData;
// }

// // Function to write a WAV header
// void writeWAVHeader(ofstream &outputFile, int dataSize, int sampleRate, int numChannels, int bitsPerSample) {
//     int byteRate = sampleRate * numChannels * bitsPerSample / 8;
//     int blockAlign = numChannels * bitsPerSample / 8;
//     int chunkSize = 36 + dataSize; // 36 + data size (since the header is always 44 bytes)
//     int subchunk2Size = dataSize;

//     // RIFF chunk descriptor
//     outputFile.write("RIFF", 4);                      // ChunkID "RIFF"
//     outputFile.write(reinterpret_cast<const char *>(&chunkSize), 4);   // ChunkSize
//     outputFile.write("WAVE", 4);                      // Format "WAVE"

//     // fmt sub-chunk
//     outputFile.write("fmt ", 4);                      // Subchunk1ID "fmt "
//     int subchunk1Size = 16;                           // Subchunk1Size (PCM)
//     outputFile.write(reinterpret_cast<const char *>(&subchunk1Size), 4);
//     int16_t audioFormat = 1;                          // AudioFormat (1 for PCM)
//     outputFile.write(reinterpret_cast<const char *>(&audioFormat), 2);
//     outputFile.write(reinterpret_cast<const char *>(&numChannels), 2); // NumChannels
//     outputFile.write(reinterpret_cast<const char *>(&sampleRate), 4);  // SampleRate
//     outputFile.write(reinterpret_cast<const char *>(&byteRate), 4);    // ByteRate
//     outputFile.write(reinterpret_cast<const char *>(&blockAlign), 2);  // BlockAlign
//     outputFile.write(reinterpret_cast<const char *>(&bitsPerSample), 2); // BitsPerSample

//     // data sub-chunk
//     outputFile.write("data", 4);                      // Subchunk2ID "data"
//     outputFile.write(reinterpret_cast<const char *>(&subchunk2Size), 4); // Subchunk2Size
// }

// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         cerr << "Usage: " << argv[0] << " <encrypted_audio_txt_file>" << endl;
//         return 1;
//     }

//     string inputFilename = argv[1];
//     string wavOutputFilename = "reconstructed_audio.wav";

//     // Open the encrypted text file containing the audio data and metadata
//     ifstream inputFile(inputFilename);
//     if (!inputFile) {
//         cerr << "Could not open the encrypted text file!" << endl;
//         return 1;
//     }

//     // Read the metadata (sample rate, channels, bit depth)
//     int sampleRate, numChannels, bitsPerSample;
//     inputFile >> sampleRate >> numChannels >> bitsPerSample;

//     // Read the remaining encrypted audio data
//     string encryptedMessage;
//     getline(inputFile, encryptedMessage, ',');  // Read the metadata line
//     getline(inputFile, encryptedMessage);       // Read the encrypted audio data

//     // Close the input file
//     inputFile.close();

//     // Decrypt the audio data
//     vector<char> decryptedAudioData = decryptAudioData(encryptedMessage);

//     // Calculate the size of the raw audio data
//     int dataSize = decryptedAudioData.size();

//     // Open a new WAV file to write the audio data back
//     ofstream outputFile(wavOutputFilename, ios::binary);
//     if (!outputFile) {
//         cerr << "Could not create the output WAV file!" << endl;
//         return 1;
//     }

//     // Write the WAV header using the metadata
//     writeWAVHeader(outputFile, dataSize, sampleRate, numChannels, bitsPerSample);

//     // Write the decrypted audio data back to the new file
//     outputFile.write(decryptedAudioData.data(), dataSize);

//     // Close the output file
//     outputFile.close();

//     cout << "Audio data decrypted and written to " << wavOutputFilename << " successfully!" << endl;

//     return 0;
// }
