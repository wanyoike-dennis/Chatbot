from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask app and model
app = Flask(__name__)
qa_model = pipeline("question-answering")

# Example FAQ data
FAQ_DATA = [
    {"question": "What is your return policy?", "answer": "You can return items within 30 days."},
    {"question": "How do I contact support?", "answer": "Email us at support@example.com."},
    {"question": "Where are you located?", "answer": "We are located in Nairobi, Kenya."},
]

# Search for the best answer in FAQ data
def find_best_answer(query):
    best_answer = {"score": 0, "answer": "Sorry, I don't know the answer to that."}
    for faq in FAQ_DATA:
        response = qa_model(question=query, context=faq["question"] + " " + faq["answer"])
        if response["score"] > best_answer["score"]:
            best_answer = response
    return best_answer["answer"]

# Endpoint for chatbot
@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "Query not provided"}), 400
    response = find_best_answer(user_query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
