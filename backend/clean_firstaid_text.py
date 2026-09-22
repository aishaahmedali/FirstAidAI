import re

INPUT_FILE = "documents/first_aid_text.txt"
OUTPUT_FILE = "documents/clean_first_aid_text.txt"

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    text = file.read()

text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)

text = re.sub(r"(?m)^\s*Wounds and bleeding\s*$", "", text)
text = re.sub(r"(?m)^\s*Introduction to First Aid\s*$", "", text)

text = re.sub(r"(?m)^\s*[0-9]\s*$", "", text)


text = text.replace("atall. all.", "at all.")
text = text.replace("X-rays, s,", "X-rays,")

text = re.sub(r"\n{3,}", "\n\n", text)

text = "\n".join(line.strip() for line in text.splitlines())

text = text.strip()

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    file.write(text)

print("Cleaned file created:")
print(OUTPUT_FILE)