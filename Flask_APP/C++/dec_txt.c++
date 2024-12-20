// #include <iostream>
// #include <string>
// #include <sstream>
// #include <cmath>
// #include <cstdlib>
// #include <ctime>
// #include <fstream>

// using namespace std;

// // Function to generate a random number between min and max (inclusive)
// int getRandomNumberInRange(int min, int max) {
//     return min + rand() % (max - min + 1);
// }

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

//         // Reconstruct the ASCII value
//         int asciiValue = (hundreds * 100) + (tens * 10) + ones;

//         // Return the corresponding character
//         return static_cast<char>(asciiValue);
//     }
// }

// // Function to perform cyclic subtraction on a digit
// int cyclicSubtraction(int digit, int cycleValue) {
//     int newDigit = digit - cycleValue;
//     if (newDigit < 0) {
//         newDigit += 10;  // Ensure non-negative digits
//     }
//     return newDigit;
// }

// // Function to reverse the cyclic randomness from the encrypted message
// string reverseRandomness(const string& encryptedMessage) {
//     // First digit is the random number
//     int randomNum = encryptedMessage[0] - '0';

//     // Initialize the cycle length and cycle value
//     int cycleLength = randomNum;
//     int cycleValue = 1;

//     stringstream reversedMessage;

//     // Start from the second character (index 1)
//     for (size_t i = 1; i < encryptedMessage.size(); ++i) {
//         int digit = encryptedMessage[i] - '0';  // Convert char to int

//         // Apply cyclic subtraction
//         int originalDigit = cyclicSubtraction(digit, cycleValue);

//         // Append the original digit to the reversed message
//         reversedMessage << originalDigit;

//         // Update cycle value, reset to 1 if it exceeds the random number
//         cycleValue = (cycleValue % cycleLength) + 1;
//     }

//     return reversedMessage.str();
// }

// // Function to decrypt a full encrypted message
// string decryptMessage(const string& encryptedMessage) {
//     stringstream decryptedMessage;

//     // Step 1: Reverse the cyclic randomness
//     string reversedMessage = reverseRandomness(encryptedMessage);

//     // Step 2: Decrypt each 5-digit block
//     size_t pos = 0;
//     size_t nextPos;

//     while (pos < reversedMessage.length()) {
//         nextPos = pos + 5;
//         if (nextPos > reversedMessage.length()) {
//             // Handle unexpected end of string
//             break;
//         }

//         string code = reversedMessage.substr(pos, 5);
//         decryptedMessage << decryptCode(code);
//         pos = nextPos;
//     }

//     return decryptedMessage.str();
// }

// // Main function for decryption
// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         cerr << "Usage: " << argv[0] << " <encrypted_filename>" << endl;
//         return 1;
//     }

//     string inputFilename = argv[1];
//     string outputFilename = "decrypted_" + inputFilename;

//     ifstream inputFile(inputFilename);

//     // Check if the file opened successfully
//     if (!inputFile.is_open()) {
//         cerr << "Error opening file: " << inputFilename << endl;
//         return 1;
//     }

//     // Read the encrypted message from the file
//     stringstream buffer;
//     buffer << inputFile.rdbuf();
//     string encryptedMessage = buffer.str();

//     // Close the input file
//     inputFile.close();

//     // Decrypt the message
//     string decryptedMessage = decryptMessage(encryptedMessage);

//     ofstream outputFile(outputFilename);

//     // Check if the output file opened successfully
//     if (!outputFile.is_open()) {
//         cerr << "Error opening file: " << outputFilename << endl;
//         return 1;
//     }

//     // Write the decrypted message to the output file
//     outputFile << decryptedMessage;

//     // Close the output file
//     outputFile.close();

//     cout << "Decrypted message saved to: " << outputFilename << endl;

//     return 0;
// }



#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <fstream>

using namespace std;

// Function to decrypt a 5-digit code
char decryptCode(const string& code) {
    int num = stoi(code);

    if (num >= 90000 && num <= 99999) {
        return ' ';  // Blank space
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
    if (newDigit < 0) {
        newDigit += 10;  // Ensure non-negative digit
    }
    return newDigit;
}

// Function to reverse the cyclic randomness from the encrypted message
string reverseRandomness(const string& encryptedMessage) {
    int randomNum = encryptedMessage[0] - '0';
    int cycleLength = randomNum;
    int cycleValue = 1;

    stringstream reversedMessage;

    for (size_t i = 1; i < encryptedMessage.size(); ++i) {
        int digit = encryptedMessage[i] - '0';
        int originalDigit = cyclicSubtraction(digit, cycleValue);
        reversedMessage << originalDigit;
        cycleValue = (cycleValue % cycleLength) + 1;
    }

    return reversedMessage.str();
}

// Function to decrypt a full encrypted message
string decryptMessage(const string& encryptedMessage) {
    stringstream decryptedMessage;

    string reversedMessage = reverseRandomness(encryptedMessage);

    size_t pos = 0;
    size_t nextPos;

    while (pos < reversedMessage.length()) {
        nextPos = pos + 5;
        if (nextPos > reversedMessage.length()) {
            break;
        }

        string code = reversedMessage.substr(pos, 5);
        decryptedMessage << decryptCode(code);
        pos = nextPos;
    }

    return decryptedMessage.str();
}

// Function to generate the output file name
string generateOutputFilename(const string& inputFilename) {
    size_t lastDot = inputFilename.find_last_of('.');
    string baseName = (lastDot == string::npos) ? inputFilename : inputFilename.substr(0, lastDot);
    return baseName + "_decrypted.txt";
}

// Main function for decryption
int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <encrypted_filename>" << endl;
        return 1;
    }

    string inputFilename = argv[1];
    string outputFilename = generateOutputFilename(inputFilename);

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

    ofstream outputFile(outputFilename);

    if (!outputFile.is_open()) {
        cerr << "Error opening file: " << outputFilename << endl;
        return 1;
    }

    outputFile << decryptedMessage;
    outputFile.close();

    cout << "Decrypted message saved to: " << outputFilename << endl;

    return 0;
}
