# fetch_and_manage_users.py

from pymongo import MongoClient, errors
import sys
import json

def get_database_connection(uri="mongodb://localhost:27017/"):
    """
    Establishes a connection to the MongoDB server.

    :param uri: MongoDB connection string.
    :return: MongoClient instance.
    """
    try:
        client = MongoClient(uri)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("Connected to MongoDB successfully.\n")
        return client
    except errors.ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
        sys.exit(1)

def fetch_all_users(collection):
    """
    Fetches all user documents with valid 'user_id's from the collection.

    :param collection: MongoDB collection.
    :return: List of user documents.
    """
    try:
        # Fetch users where 'user_id' exists and is not in the invalid list
        users = list(collection.find(
            { "user_id": { "$exists": True, "$nin": ["N/A", ""] } },  # Use $nin for multiple invalid values
            { "_id": 0, "user_id": 1 }  # Fetch only 'user_id'
        ))
        if users:
            print(f"Fetched {len(users)} user(s):\n")
            for idx, user in enumerate(users, start=1):
                user_id = user.get('user_id', 'N/A')
                print(f"{idx}. User ID: {user_id}")
        else:
            print("No users found in the collection.")
        return users
    except Exception as e:
        print(f"An error occurred while fetching users: {e}")
        return []

def fetch_user_details(collection, user_id):
    """
    Fetches and prints the structure of a user's data with True/False indicating field presence.

    :param collection: MongoDB collection.
    :param user_id: The 'user_id' of the user.
    """
    try:
        user = collection.find_one({"user_id": user_id}, {"_id": 0})
        if user:
            print(f"\nStructure of User '{user_id}':")
            structured_data = create_structure_with_booleans(user)
            # Pretty-print the structured data
            print(json.dumps(structured_data, indent=4))
            print("")  # Add a newline for better readability
        else:
            print(f"No details found for user '{user_id}'.\n")
    except Exception as e:
        print(f"An error occurred while fetching details for user '{user_id}': {e}\n")

def create_structure_with_booleans(data):
    """
    Recursively creates a structure with True/False indicating the presence of each field's value.

    :param data: The user document (dict) from MongoDB.
    :return: A nested dictionary with True/False values.
    """
    if isinstance(data, dict):
        return { key: create_structure_with_booleans(value) for key, value in data.items() }
    elif isinstance(data, list):
        # For lists, return True if non-empty, False if empty
        return bool(data)
    else:
        # For non-dict and non-list types, return True if value is truthy, else False
        return bool(data)

def delete_user_by_index(collection, users, index):
    """
    Deletes a user based on their index in the users list.

    :param collection: MongoDB collection.
    :param users: List of user documents.
    :param index: The index number of the user to delete (1-based).
    :return: Boolean indicating success or failure.
    """
    if 1 <= index <= len(users):
        user = users[index - 1]
        user_id = user.get('user_id')
        if user_id and user_id != 'N/A':
            try:
                confirmation = input(f"Are you sure you want to delete user '{user_id}'? (yes/no): ").strip().lower()
                if confirmation == 'yes':
                    result = collection.delete_one({"user_id": user_id})
                    if result.deleted_count > 0:
                        print(f"Successfully deleted user '{user_id}'.\n")
                        return True
                    else:
                        print(f"Failed to delete user '{user_id}'.\n")
                        return False
                else:
                    print("Deletion canceled.\n")
                    return False
            except Exception as e:
                print(f"An error occurred while deleting user '{user_id}': {e}\n")
                return False
        else:
            print(f"User at index {index} has no valid 'user_id'. Cannot delete.\n")
            return False
    else:
        print(f"Invalid index: {index}. Please enter a number between 1 and {len(users)}.\n")
        return False

def delete_all_users(collection):
    """
    Deletes all users from the collection after confirmation.

    :param collection: MongoDB collection.
    :return: Boolean indicating success or failure.
    """
    try:
        confirmation = input("Are you sure you want to delete **ALL** users? This action cannot be undone. (yes/no): ").strip().lower()
        if confirmation == 'yes':
            result = collection.delete_many({ "user_id": { "$exists": True, "$nin": ["N/A", ""] } })
            print(f"Successfully deleted {result.deleted_count} user(s).\n")
            return True
        else:
            print("Deletion of all users canceled.\n")
            return False
    except Exception as e:
        print(f"An error occurred while deleting all users: {e}\n")
        return False


def fetch_user_questions_and_details(collection, user_id):
    """
    Fetches and displays detailed questions, transcriptions, and recordings for a specific user.

    :param collection: MongoDB collection.
    :param user_id: The 'user_id' of the user.
    """
    try:
        user = collection.find_one({"user_id": user_id}, {"_id": 0, "answers": 1})
        if user and "answers" in user:
            print(f"\nDetailed Questions and Answers for User '{user_id}':")
            for question, details in user["answers"].items():
                question_text = details.get("question_text", "N/A")
                transcription = details.get("transcription", "N/A")
                recordings = details.get("recordings", [])
                
                print(f"Question: {question_text}")
                print(f"Transcription: {transcription}")
                
                if recordings:
                    print("Recordings:")
                    for recording in recordings:
                        recording_id = recording.get("recording_id", "N/A")
                        recording_transcription = recording.get("transcription", "N/A")
                        embedding = recording.get("embedding", "N/A")
                        print(f"  - Recording ID: {recording_id}")
                        print(f"    Transcription: {recording_transcription}")
                        print(f"    Embedding: {embedding}")
                else:
                    print("No recordings available for this question.")
                print("")  # Add spacing for readability
        else:
            print(f"No questions and details found for user '{user_id}'.\n")
    except Exception as e:
        print(f"An error occurred while fetching questions and details for user '{user_id}': {e}\n")

def show_database_structure(database):
    """
    Displays the structure of the database, including collections and their sample documents.

    :param database: MongoDB database object.
    """
    try:
        print("\nDatabase Structure:\n")
        collections = database.list_collection_names()
        if not collections:
            print("No collections found in the database.\n")
            return

        for idx, collection_name in enumerate(collections, start=1):
            print(f"{idx}. Collection: {collection_name}")
            collection = database[collection_name]
            # Fetch a sample document from the collection
            sample_document = collection.find_one({}, {"_id": 0})
            if sample_document:
                print(f"   Sample Document: {json.dumps(sample_document, indent=4)}")
            else:
                print("   No documents found in this collection.")
            print("")  # Add a newline for better readability
    except Exception as e:
        print(f"An error occurred while fetching the database structure: {e}\n")

def main():
    # MongoDB connection URI
    mongo_uri = "mongodb://localhost:27017/"  # Update if necessary

    # Connect to MongoDB
    client = get_database_connection(mongo_uri)

    # Access the database
    db = client.voice_authentication_system

    # Access the 'users' collection
    users_collection = db.users

    while True:
        # Fetch and display all users
        users = fetch_all_users(users_collection)

        if not users:
            break  # Exit if no users are found

        # Provide options to the user
        print("\nOptions:")
        print("1. Delete a user by index")
        print("2. Delete **ALL** users")
        print("3. View user structure by index")
        print("4. View questions and answers by index")
        print("5. Show database structure")
        print("6. Refresh user list")
        print("7. Exit")

        choice = input("Enter your choice (1/2/3/4/5/6/7): ").strip()

        if choice == '1':
            try:
                index_input = input(f"\nEnter the user index to delete (1-{len(users)}): ").strip()
                index = int(index_input)
                delete_user_by_index(users_collection, users, index)
            except ValueError:
                print("Invalid input. Please enter a valid number.\n")
        elif choice == '2':
            delete_all_users(users_collection)
        elif choice == '3':
            try:
                index_input = input(f"\nEnter the user index to view structure (1-{len(users)}): ").strip()
                index = int(index_input)
                if 1 <= index <= len(users):
                    user = users[index - 1]
                    user_id = user.get('user_id')
                    if user_id and user_id != 'N/A':
                        fetch_user_details(users_collection, user_id)
                    else:
                        print(f"User at index {index} has no valid 'user_id'. Cannot display structure.\n")
                else:
                    print(f"Invalid index: {index}. Please enter a number between 1 and {len(users)}.\n")
            except ValueError:
                print("Invalid input. Please enter a valid number.\n")
        elif choice == '4':
            try:
                index_input = input(f"\nEnter the user index to view detailed questions and answers (1-{len(users)}): ").strip()
                index = int(index_input)
                if 1 <= index <= len(users):
                    user = users[index - 1]
                    user_id = user.get('user_id')
                    if user_id and user_id != 'N/A':
                        fetch_user_questions_and_details(users_collection, user_id)
                    else:
                        print(f"User at index {index} has no valid 'user_id'. Cannot display details.\n")
                else:
                    print(f"Invalid index: {index}. Please enter a number between 1 and {len(users)}.\n")
            except ValueError:
                print("Invalid input. Please enter a valid number.\n")
        elif choice == '5':
            show_database_structure(db)
        elif choice == '6':
            print("\nRefreshing user list...\n")
            continue  # Loop again to refresh
        elif choice == '7':
            print("\nExiting the script.")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5, 6, or 7.\n")

    # Close the connection
    client.close()
    print("MongoDB connection closed.")

if __name__ == "__main__":
    main()
