#include <iostream>
#include <fstream>   // For file input/output
#include <sstream>   // For stringstream
#include <locale>    // For locale and codecvt
#include <codecvt>   // For codecvt_utf8
#include <cstdlib>   // For rand()
#include <ctime>     // For time()
#include <chrono>    // For measuring time
#include <unordered_map>  // For memoization

using namespace std;
using namespace std::chrono;

// Function to compute the encrypted value based on ASCII value
int encryptChar(int asciiValue) {
    int ones = asciiValue % 10;
    int tens = (asciiValue / 10) % 10;
    int hundreds = (asciiValue / 100) % 10;

    int result = (ones * ones) + (tens * tens * tens) + (hundreds * hundreds);
    return result;
}

// Function to calculate the two-digit differentiator
string calculateDifferentiator(int asciiValue) {
    int ones = asciiValue % 10;
    int tens = (asciiValue / 10) % 10;
    int hundreds = (asciiValue / 100) % 10;
    int differentiator = (hundreds * 10) + ones;

    // Ensure it's always two digits by adding a leading zero if needed
    stringstream ss;
    ss << (differentiator < 10 ? "0" : "") << differentiator;

    return ss.str();
}

// Function to generate a random number between min and max (inclusive), excluding a specific number
int getRandomNumberExcluding(int min, int max, int exclude) {
    int randomNum;
    do {
        randomNum = min + rand() % (max - min + 1);
    } while (randomNum == exclude);
    return randomNum;
}

int getRandomNumberInRange(int min, int max) {
    return min + rand() % (max - min + 1);
}
// Function to handle the encryption of a string with memoization
string encryptAndMapToChar(const string& input) {
    stringstream encryptedMessage; // Use stringstream to build the result
    srand(time(0)); // Seed for randomness

    // Memoization map to store previously encrypted characters
    unordered_map<wchar_t, string> memo;

    // Convert input to wstring to handle wide characters
    wstring winput = wstring_convert<codecvt_utf8<wchar_t>>().from_bytes(input);

    for (wchar_t wc : winput) {
        if (memo.find(wc) != memo.end()) {
            // If character is already encrypted, use the stored result
            encryptedMessage << memo[wc];
        } else {
            // Encrypt the character and store the result in the memo
            int asciiValue = static_cast<int>(wc);

            if (wc == L' ') {
                // Handle blank space with a random number between 90000 and 99999
                int randomSpaceValue = getRandomNumberInRange(90000, 99999);
                stringstream ss;
                ss << randomSpaceValue;
                memo[wc] = ss.str();  // Store the result in memo
                encryptedMessage << memo[wc];
            } else if (asciiValue > 255) {
                // Handle non-ASCII characters explicitly
                memo[wc] = "INVALID";  // Store in memo
                encryptedMessage << memo[wc];
            } else {
                int encryptedValue = encryptChar(asciiValue);
                string differentiator = calculateDifferentiator(asciiValue);

                // Convert encrypted value to string and pad with leading zeros to ensure 3 digits
                stringstream encryptedValueStr;
                encryptedValueStr << (encryptedValue < 100 ? (encryptedValue < 10 ? "00" : "0") : "") << encryptedValue;

                // Append the final 5-digit encrypted result
                string result = encryptedValueStr.str() + differentiator;
                memo[wc] = result;  // Store the result in memo
                encryptedMessage << result;
            }
        }
    }

    return encryptedMessage.str(); // Return the result as a string
}

// Function to apply randomness to the encrypted message
string applyRandomness(const string& encryptedMessage) {
    // Generate random number between 1-9, excluding 5
    int randomNum = getRandomNumberExcluding(1, 9, 5);
    
    // Start the result string with the random number
    stringstream randomizedMessage;
    randomizedMessage << randomNum;

    int cycleLength = randomNum;
    int cycleValue = 1;

    // Apply cyclic addition to the digits starting from index 1
    for (size_t i = 0; i < encryptedMessage.size(); ++i) {
        int digit = encryptedMessage[i] - '0';  // Convert char to int

        // Add cycle value to the digit
        int newDigit = (digit + cycleValue) % 10;

        // Append the modified digit to the randomized message
        randomizedMessage << newDigit;

        // Update cycle value, reset to 1 if it exceeds the random number
        cycleValue = (cycleValue % cycleLength) + 1;
    }

    return randomizedMessage.str(); // Return the randomized message
}

// Function to read the contents of a text file into a string
string readFile(const string& filename) {
    ifstream file(filename);
    if (!file) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1); // Exit if file can't be opened
    }

    stringstream buffer;
    buffer << file.rdbuf();  // Read the entire file into a stringstream
    file.close();

    return buffer.str();     // Return the file contents as a string
}

// Function to write the encrypted message to a text file
void writeToFile(const string& filename, const string& data) {
    ofstream file(filename);
    if (!file) {
        cerr << "Unable to open file: " << filename << endl;
        exit(1); // Exit if file can't be opened
    }

    file << data; // Write the encrypted data to the file
    file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <input_filename>" << endl;
        return 1;
    }

    string inputFilename = argv[1];
    string outputFilename = "encrypted_" + inputFilename;

    // Read the file contents
    string message = readFile(inputFilename);

    // Start measuring time
    auto start = high_resolution_clock::now();

    // Step 1: Encrypt the contents of the file with memoization
    string encryptedMessage = encryptAndMapToChar(message);

    // Step 2: Apply randomness to the encrypted message
    string randomizedMessage = applyRandomness(encryptedMessage);

    // Stop measuring time
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    // Write the randomized encrypted message to the output file
    writeToFile(outputFilename, randomizedMessage);
    
    // Output the time taken for encryption
    cout << "Encryption completed in " << duration.count() << " milliseconds" << endl;
    cout << "Encrypted message saved to: " << outputFilename << endl;

    return 0;
}