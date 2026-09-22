from old_rag.firstaid_ai import ask_question

test_questions = [
    "Someone is choking and cannot breathe.",
    "How do I treat a burn?",
    "Someone is bleeding heavily.",
    "How do I help someone having a seizure?",
    "What should I do if someone faints?",
    "Someone has a broken arm.",
    "What should I do if someone is drowning?",
    "How do I treat a snake bite?",
    "How do I perform CPR?",
    "Who won the World Cup?"
]

for i, question in enumerate(test_questions, start=1):
    print("=" * 70)
    print(f"Test {i}")
    print("Question:", question)
    print("-" * 70)

    answer = ask_question(question)

    print(answer)
    print()