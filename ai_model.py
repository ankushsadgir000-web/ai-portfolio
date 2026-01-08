import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load sentences
def load_data(folder="data"):
    sentences = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text = f.read().lower()
                for line in text.split("."):
                    line = line.strip()
                    if line:
                        sentences.append(line)
    return sentences


sentences = load_data()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(sentences)


# Keyword routing (IMPORTANT)
def keyword_match(question):
    q = question.lower()

    if "about" in q or "yourself" in q or "who are you" in q:
        return "about"
    if "skill" in q:
        return "skills:"
    if "project" in q:
        return "projects:"
    if "education" in q or "study" in q:
        return "education"
    return None


def ask_ai(question):
    key = keyword_match(question)

    # If keyword found, return matching sentence
    if key:
        for s in sentences:
            if key in s:
                return s

    # Else fallback to TF-IDF
    question_vector = vectorizer.transform([question.lower()])
    similarities = cosine_similarity(question_vector, tfidf_matrix)

    best_index = similarities.argmax()
    best_score = similarities[0][best_index]

    if best_score < 0.05:
        return "Sorry, I don't have information related to that."

    return sentences[best_index]


if __name__ == "__main__":
    print("AI Assistant: Ask about Ankush (type 'exit' to quit)\n")

    while True:
        q = input("You: ")
        if q.lower() == "exit":
            print("AI Assistant: Goodbye!")
            break

        print("\nAI Assistant:", ask_ai(q), "\n")
