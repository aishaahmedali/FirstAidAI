from old_rag.firstaid_ai import ask_question

print("Welcome to the AI Powered First Aid Assistant!")
print("This system answers first aid questions using the First Aid knowledge base.")

while True:
    print("\n" + "=" * 50)
    print("        AI Powered First Aid Assistant")
    print("=" * 50)
    print("1. Ask a First Aid Question")
    print("2. Exit")
    print("=" * 50)

    choice = input("Choose an option: ")

    if choice == "1":
        question = input("\nEnter your first aid question: ")

        if not question.strip():
            print("Please enter a valid question.")
            continue

        print("\nGenerating response...\n")

        answer = ask_question(question)

        print("=" * 50)
        print("FirstAidAI Response")
        print("=" * 50)
        print(answer)

    elif choice == "2":
        print("\nThank you for using FirstAidAI.")
        print("Stay safe!")
        break

    else:
        print("\nInvalid option. Please try again.")