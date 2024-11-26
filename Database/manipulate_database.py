from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)

# Set up MongoDB connection (adjust the URI if needed)
client = MongoClient('mongodb://localhost:27017/')
db = client['user_database']
users_collection = db['users']

# Home page to show the users
@app.route('/')
def index():
    users = list(users_collection.find())
    user_count = users_collection.count_documents({})
    return render_template('index.html', users=users, user_count=user_count)

# View specific user details
@app.route('/user/<user_id>')
def view_user(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return "User not found!", 404
    return render_template('view_user.html', user=user)

# Route to delete a user
@app.route('/delete_user/<user_id>', methods=['POST'])
def delete_user(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return "User not found!", 404
    users_collection.delete_one({"_id": ObjectId(user_id)})
    return redirect(url_for('index'))

# Route to update a user
@app.route('/update_user/<user_id>', methods=['GET', 'POST'])
def update_user(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return "User not found!", 404

    if request.method == 'POST':
        # Get the updated details from the form
        new_data = {
            'userid': request.form['userid'],
            'question_1': request.form['question_1'],
            'answer_1': request.form['answer_1'],
            'question_2': request.form['question_2'],
            'answer_2': request.form['answer_2'],
            'question_3': request.form['question_3'],
            'answer_3': request.form['answer_3'],
        }
        # Update the user in the database
        users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": new_data}
        )
        return redirect(url_for('view_user', user_id=user_id))

    return render_template('update_user.html', user=user)

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)
