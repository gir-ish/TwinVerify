from pymongo import MongoClient

# Connect to MongoDB server
client = MongoClient("mongodb://localhost:27017/")

# Access a database
db = client["example_db"]

# Access a collection
collection = db["users"]

def create_user():
    username = input("Enter your username: ")
    
    # Ask for number of security questions
    num_questions = int(input("How many security questions would you like to add? (e.g., 3): "))
    
    questions = []
    for i in range(num_questions):
        question = input(f"Enter question {i+1}: ")
        answer = input(f"Enter the answer for question {i+1}: ")
        questions.append({"question": question, "answer": answer})
    
    # Create user document
    user_data = {
        "username": username,
        "security_questions": questions
    }
    
    # Insert user into the database
    collection.insert_one(user_data)
    print(f"User '{username}' created successfully!")

def update_user():
    username = input("Enter the username of the account you want to update: ")
    
    # Check if the user exists
    user = collection.find_one({"username": username})
    if user:
        print(f"Updating security questions for user: {username}")
        
        # Ask for the number of questions to update
        num_questions = int(input("How many questions would you like to update? "))
        
        # Get updated answers
        updated_questions = []
        for i in range(num_questions):
            question = input(f"Enter question {i+1} to update: ")
            answer = input(f"Enter new answer for question {i+1}: ")
            updated_questions.append({"question": question, "answer": answer})
        
        # Update the user document in the collection
        collection.update_one(
            {"username": username},
            {"$set": {"security_questions": updated_questions}}
        )
        print(f"Security questions for user '{username}' updated successfully!")
    else:
        print(f"User '{username}' not found!")

def delete_user():
    username = input("Enter the username of the account you want to delete: ")
    
    # Check if the user exists
    user = collection.find_one({"username": username})
    if user:
        # Delete the user from the collection
        collection.delete_one({"username": username})
        print(f"User '{username}' deleted successfully!")
    else:
        print(f"User '{username}' not found!")

def display_users():
    print("Users in the database:")
    users = collection.find()
    for user in users:
        print(f"Username: {user['username']}")
        if "security_questions" in user:
            print("Security Questions:")
            for sq in user["security_questions"]:
                print(f"  Question: {sq['question']}, Answer: {sq['answer']}")
        print("-" * 50)

# Menu options
while True:
    print("\n1. Create User")
    print("2. Update User")
    print("3. Delete User")
    print("4. Display All Users")
    print("5. Exit")
    
    choice = input("Choose an option (1-5): ")
    
    if choice == "1":
        create_user()
    elif choice == "2":
        update_user()
    elif choice == "3":
        delete_user()
    elif choice == "4":
        display_users()
    elif choice == "5":
        break
    else:
        print("Invalid option. Please try again.")
