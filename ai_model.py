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
    q = question.lower().strip()

    # Greetings
    if q in ["hi", "hello", "hey"]:
        return "Hello! 👋 You can ask me about my skills, projects, or education."

    if "thank" in q:
        return "You're welcome! 😊 Feel free to ask more."

    # Projects
    if "project" in q:
        project_lines = [
            s for s in sentences
            if "portfolio" in s or "system" in s or "security" in s
        ]
        return "Here are my projects:\n" + "\n".join(project_lines)

    # Skills
    if "skill" in q:
        skill_lines = [
            s for s in sentences
            if "programming" in s or "web" in s or "tools" in s
        ]
        return "My skills include:\n" + "\n".join(skill_lines)

    # Education
    if "education" in q or "study" in q:
        edu_lines = [
            s for s in sentences
            if "engineering" in s or "education" in s
        ]
        return "Education details:\n" + "\n".join(edu_lines)

    # Fallback (TF-IDF)
    question_vector = vectorizer.transform([q])
    similarities = cosine_similarity(question_vector, tfidf_matrix)
    best_index = similarities.argmax()

    return "Here’s what I found:\n" + sentences[best_index]

