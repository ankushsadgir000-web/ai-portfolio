from flask import Flask, request, jsonify, render_template
from ai_model import ask_ai

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    answer = ask_ai(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
