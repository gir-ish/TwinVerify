// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <sstream>
// #include <cstdlib>   // For rand()
// #include <ctime>     // For time()
// #include <chrono>    // For measuring time
// #include <unordered_map>  // For memoization
// #include <opencv2/opencv.hpp>  // For handling images

// using namespace std;
// using namespace std::chrono;
// using namespace cv;

// // Function to compute the encrypted value based on a byte value
// int encryptByte(int byteValue) {
//     int ones = byteValue % 10;
//     int tens = (byteValue / 10) % 10;
//     int hundreds = (byteValue / 100) % 10;

//     int result = (ones * ones) + (tens * tens * tens) + (hundreds * hundreds);
//     return result;
// }

// // Function to calculate the two-digit differentiator for encryption
// string calculateDifferentiator(int byteValue) {
//     int ones = byteValue % 10;
//     int tens = (byteValue / 10) % 10;
//     int hundreds = (byteValue / 100) % 10;
//     int differentiator = (hundreds * 10) + ones;

//     // Ensure it's always two digits by adding a leading zero if needed
//     stringstream ss;
//     ss << (differentiator < 10 ? "0" : "") << differentiator;

//     return ss.str();
// }

// // Function to handle the encryption of image data with memoization
// string encryptAndMapToChar(const vector<unsigned char>& pixelData) {
//     stringstream encryptedMessage; // Use stringstream to build the result
//     srand(time(0)); // Seed for randomness

//     // Memoization map to store previously encrypted characters
//     unordered_map<int, string> memo;

//     for (unsigned char byte : pixelData) {
//         int byteValue = static_cast<int>(byte);

//         if (memo.find(byteValue) != memo.end()) {
//             // If byte is already encrypted, use the stored result
//             encryptedMessage << memo[byteValue];
//         } else {
//             // Encrypt the byte and store the result in the memo
//             int encryptedValue = encryptByte(byteValue);
//             string differentiator = calculateDifferentiator(byteValue);

//             // Convert encrypted value to string and pad with leading zeros to ensure 3 digits
//             stringstream encryptedValueStr;
//             encryptedValueStr << (encryptedValue < 100 ? (encryptedValue < 10 ? "00" : "0") : "") << encryptedValue;

//             // Append the final 5-digit encrypted result
//             string result = encryptedValueStr.str() + differentiator;
//             memo[byteValue] = result;  // Store the result in memo
//             encryptedMessage << result;
//         }
//     }

//     return encryptedMessage.str(); // Return the result as a string
// }

// // Function to write both metadata and encrypted data to a text file
// void writeEncryptedToFile(const string& filename, const string& metadata, const string& encryptedData) {
//     ofstream file(filename);
//     if (!file) {
//         cerr << "Unable to open file: " << filename << endl;
//         exit(1); // Exit if file can't be opened
//     }

//     // Write metadata and encrypted data to the file
//     file << metadata << "," << encryptedData;
//     file.close();
// }

// // Function to convert an image to a text format and encrypt the pixel data
// void convertAndEncryptImage(const string& inputFilename, const string& outputFilename) {
//     // Load the image
//     Mat image = imread(inputFilename, IMREAD_COLOR);
//     if (image.empty()) {
//         cerr << "Could not open or find the image: " << inputFilename << endl;
//         exit(1);
//     }

//     // Extract image dimensions and other metadata
//     int rows = image.rows;
//     int cols = image.cols;
//     int channels = image.channels();

//     // Store metadata in a string
//     stringstream metadataStream;
//     metadataStream << rows << " " << cols << " " << channels;
//     string metadata = metadataStream.str();

//     // Flatten the image into a 1D vector of pixel data
//     vector<unsigned char> pixelData;
//     pixelData.assign(image.datastart, image.dataend);

//     // Encrypt the pixel data
//     string encryptedData = encryptAndMapToChar(pixelData);

//     // Write the metadata and encrypted pixel data to the output file
//     writeEncryptedToFile(outputFilename, metadata, encryptedData);

//     cout << "Encryption completed and saved to: " << outputFilename << endl;
// }

// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         cerr << "Usage: " << argv[0] << " <input_image_filename>" << endl;
//         return 1;
//     }

//     string inputFilename = argv[1];
//     string outputFilename = "encrypted_image_data.txt";

//     // Start measuring time
//     auto start = high_resolution_clock::now();

//     // Convert and encrypt the image
//     convertAndEncryptImage(inputFilename, outputFilename);

//     // Stop measuring time
//     auto end = high_resolution_clock::now();
//     auto duration = duration_cast<milliseconds>(end - start);

//     cout << "Encryption completed in " << duration.count() << " milliseconds" << endl;

//     return 0;
// }
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdlib>   // For rand()
#include <ctime>     // For time()
#include <chrono>    // For measuring time
#include <unordered_map>  // For memoization
#include <opencv2/opencv.hpp>  // For handling images

using namespace std;
using namespace std::chrono;
using namespace cv;

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

// Function to handle the encryption of image data with memoization
string encryptAndMapToChar(const vector<unsigned char>& pixelData) {
    stringstream encryptedMessage; // Use stringstream to build the result
    srand(time(0)); // Seed for randomness

    // Memoization map to store previously encrypted characters
    unordered_map<int, string> memo;

    for (unsigned char byte : pixelData) {
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

// Function to generate the encrypted filename based on the input filename
string generateEncryptedFilename(const string& inputFilename) {
    // Extract the base name with extension
    size_t lastSlash = inputFilename.find_last_of("/\\");
    string baseName = (lastSlash == string::npos) ? inputFilename : inputFilename.substr(lastSlash + 1);

    // Append .img to the base name, retaining the original extension
    return "encrypted_" + baseName + ".img";
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

// Function to convert an image to a text format and encrypt the pixel data
void convertAndEncryptImage(const string& inputFilename, const string& outputFilename) {
    // Load the image
    Mat image = imread(inputFilename, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Could not open or find the image: " << inputFilename << endl;
        exit(1);
    }

    // Extract image dimensions and other metadata
    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();

    // Store metadata in a string
    stringstream metadataStream;
    metadataStream << rows << " " << cols << " " << channels;
    string metadata = metadataStream.str();

    // Flatten the image into a 1D vector of pixel data
    vector<unsigned char> pixelData;
    pixelData.assign(image.datastart, image.dataend);

    // Encrypt the pixel data
    string encryptedData = encryptAndMapToChar(pixelData);

    // Write the metadata and encrypted pixel data to the output file
    writeEncryptedToFile(outputFilename, metadata, encryptedData);

    cout << "Encryption completed and saved to: " << outputFilename << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_image_filename>" << endl;
        return 1;
    }

    string inputFilename = argv[1];
    string outputFilename = generateEncryptedFilename(inputFilename);

    // Start measuring time
    auto start = high_resolution_clock::now();

    // Convert and encrypt the image
    convertAndEncryptImage(inputFilename, outputFilename);

    // Stop measuring time
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Encryption completed in " << duration.count() << " milliseconds" << endl;

    return 0;
}
