#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <fstream>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

// Function to decrypt a 5-digit code
char decryptCode(const string& code) {
    int num = stoi(code);

    if (num >= 90000 && num <= 99999) {
        return ' ';
    } else {
        int encryptedValue = stoi(code.substr(0, 3));
        int hundreds = stoi(code.substr(3, 1));
        int ones = stoi(code.substr(4, 1));

        int sumOfSquares = (hundreds * hundreds) + (ones * ones);
        int difference = encryptedValue - sumOfSquares;

        int tens = round(cbrt(difference));
        int asciiValue = (hundreds * 100) + (tens * 10) + ones;

        return static_cast<char>(asciiValue);
    }
}

// Function to perform cyclic subtraction on a digit
int cyclicSubtraction(int digit, int cycleValue) {
    int newDigit = digit - cycleValue;
    return (newDigit < 0) ? (newDigit + 10) : newDigit;
}

// Function to reverse cyclic randomness
string reverseRandomness(const string& encryptedMessage) {
    int randomNum = encryptedMessage[0] - '0';
    int cycleValue = 1;

    stringstream reversedMessage;
    for (size_t i = 1; i < encryptedMessage.size(); ++i) {
        int digit = encryptedMessage[i] - '0';
        int originalDigit = cyclicSubtraction(digit, cycleValue);
        reversedMessage << originalDigit;

        cycleValue = (cycleValue % randomNum) + 1;
    }
    return reversedMessage.str();
}

// Function to extract the arguments from the decrypted text
void extractArguments(string& decryptedMessage, string& arg1, string& arg2, string& arg3) {
    stringstream ss(decryptedMessage);
    getline(ss, arg1, ' ');
    getline(ss, arg2, ' ');
    getline(ss, arg3, ' ');

    // Remove the extracted arguments from the decryptedMessage
    decryptedMessage = decryptedMessage.substr(arg1.length() + arg2.length() + arg3.length() + 3); // +3 for spaces
}

// Function to decrypt a message
string decryptMessage(const string& encryptedMessage) {
    stringstream decryptedMessage;

    string reversedMessage = reverseRandomness(encryptedMessage);

    size_t pos = 0;
    while (pos < reversedMessage.length()) {
        string code = reversedMessage.substr(pos, 5);
        decryptedMessage << decryptCode(code);
        pos += 5;
    }

    return decryptedMessage.str();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <encrypted_filename>" << endl;
        return 1;
    }

    string inputFilename = argv[1];
    if (!fs::exists(inputFilename)) {
        cerr << "Error: Input file does not exist: " << inputFilename << endl;
        return 1;
    }

    string outputFilename = fs::path(inputFilename).parent_path() / ("decrypted_" + fs::path(inputFilename).filename().string());

    ifstream inputFile(inputFilename);
    if (!inputFile.is_open()) {
        cerr << "Error opening file: " << inputFilename << endl;
        return 1;
    }

    stringstream buffer;
    buffer << inputFile.rdbuf();
    string encryptedMessage = buffer.str();
    inputFile.close();

    string decryptedMessage = decryptMessage(encryptedMessage);

    string arg1, arg2, arg3;
    extractArguments(decryptedMessage, arg1, arg2, arg3);

    cout << "Extracted Arguments: " << arg1 << ", " << arg2 << ", " << arg3 << endl;

    ofstream outputFile(outputFilename);
    if (!outputFile.is_open()) {
        cerr << "Error opening file for writing: " << outputFilename << endl;
        return 1;
    }

    outputFile << decryptedMessage;
    outputFile.close();

    cout << "Decrypted file saved to: " << outputFilename << endl;
    return 0;
}
