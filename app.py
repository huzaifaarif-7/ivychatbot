from flask import Flask, request, jsonify, render_template

from internetworks import chat_with_bot


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    bot_response = chat_with_bot(user_message)
    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(debug=True)



