from flask import Flask, request, jsonify, render_template
from firstaid_system import ask_firstaid

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():

    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({
            "error": "Please provide a question."
        }), 400

    question = data["question"].strip()

    if not question:
        return jsonify({
            "error": "Please enter a valid question."
        }), 400

    try:

        result = ask_firstaid(question)

        return jsonify({
            "question": question,
            "category": result["category"],
            "confidence": result["confidence"],
            "retrieved_context": result.get("context", ""),
            "answer": result["answer"]
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )