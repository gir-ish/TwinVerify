#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdlib>   // For rand()
#include <ctime>     // For time()
#include <chrono>    // For measuring time
#include <opencv2/opencv.hpp>  // For handling images

using namespace std;
using namespace std::chrono;
using namespace cv;

// Function to reverse the encrypted value based on the original encryption logic
int decryptByte(const string& encryptedValueStr, const string& differentiatorStr) {
    int encryptedValue = stoi(encryptedValueStr);  // Convert encrypted string to int
    int differentiator = stoi(differentiatorStr);  // Convert differentiator string to int

    // Reverse the encryption logic to find the original hundreds and ones places
    int hundreds = differentiator / 10;
    int ones = differentiator % 10;

    // Try to find a byteValue that satisfies the encryption logic
    for (int byteValue = 0; byteValue <= 255; ++byteValue) {
        int currentOnes = byteValue % 10;
        int currentTens = (byteValue / 10) % 10;
        int currentHundreds = (byteValue / 100) % 10;

        if (currentOnes == ones && currentHundreds == hundreds) {
            int result = (currentOnes * currentOnes) + (currentTens * currentTens * currentTens) + (currentHundreds * currentHundreds);
            if (result == encryptedValue) {
                return byteValue;
            }
        }
    }

    return -1;  // Return -1 if no matching byteValue is found
}

// Function to decrypt the encrypted data and recover the pixel values
vector<unsigned char> decryptAndMapToPixels(const string& encryptedData) {
    vector<unsigned char> pixelData; // Vector to hold the decrypted pixel values

    for (size_t i = 0; i < encryptedData.size(); i += 5) {
        // Extract the 3-digit encrypted value and 2-digit differentiator
        string encryptedValueStr = encryptedData.substr(i, 3);
        string differentiatorStr = encryptedData.substr(i + 3, 2);

        // Decrypt the byte value
        int byteValue = decryptByte(encryptedValueStr, differentiatorStr);

        if (byteValue != -1) {
            // Add the decrypted byte value to the pixel data vector
            pixelData.push_back(static_cast<unsigned char>(byteValue));
        } else {
            cerr << "Decryption error: could not match encrypted value." << endl;
            exit(1);
        }
    }

    return pixelData;
}

// Function to write the decrypted pixel data to an image file
void writeDecryptedImage(const vector<unsigned char>& pixelData, int rows, int cols, int channels, const string& outputFilename) {
    // Create an empty Mat object to store the image data
    Mat image(rows, cols, (channels == 3 ? CV_8UC3 : CV_8UC1));

    // Copy the pixel data into the Mat object
    memcpy(image.data, pixelData.data(), pixelData.size());

    // Save the image to a file
    imwrite(outputFilename, image);

    cout << "Decrypted image saved to: " << outputFilename << endl;
}

// Function to read the encrypted data and metadata from the text file
void decryptImageFile(const string& inputFilename, const string& outputFilename) {
    ifstream inputFile(inputFilename);
    if (!inputFile) {
        cerr << "Could not open the file: " << inputFilename << endl;
        exit(1);
    }

    // Read the metadata and encrypted data
    string line;
    getline(inputFile, line);

    // Split the metadata and encrypted data
    size_t commaPos = line.find(',');
    string metadata = line.substr(0, commaPos);
    string encryptedData = line.substr(commaPos + 1);

    // Extract the metadata (rows, cols, channels)
    stringstream metadataStream(metadata);
    int rows, cols, channels;
    metadataStream >> rows >> cols >> channels;

    // Decrypt the pixel data
    vector<unsigned char> pixelData = decryptAndMapToPixels(encryptedData);

    // Write the decrypted pixel data to an image file
    writeDecryptedImage(pixelData, rows, cols, channels, outputFilename);

    inputFile.close();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_encrypted_filename>" << endl;
        return 1;
    }

    string inputFilename = argv[1];
    string outputFilename = "decrypted_image.png";

    // Start measuring time
    auto start = high_resolution_clock::now();

    // Decrypt the image file and save the decrypted image
    decryptImageFile(inputFilename, outputFilename);

    // Stop measuring time
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Decryption completed in " << duration.count() << " milliseconds" << endl;

    return 0;
}
