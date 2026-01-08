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

    # Greeting
    if q.startswith(("hi", "hello", "hey")):
        return (
            "👋 Hello!\n\n"
            "I’m Ankush’s AI Portfolio Assistant.\n"
            "You can ask me about:\n"
            "• About Me\n"
            "• Skills\n"
            "• Projects\n"
            "• Education"
        )

    # About Me
    if "about" in q or "yourself" in q or "who are you" in q:
        return (
            "📌 **About Me**\n\n"
            "• Full Name: Ankush Vinod Shardul\n"
            "• Education: Second-year Computer Engineering student\n"
            "• Interests: Software Development, AI, Problem Solving\n"
            "• Strengths: Programming, Databases, Networking, OS\n"
            "• Goal: Seeking internship opportunities for industry exposure"
        )

    # Skills
    if "skill" in q:
        return (
            "🛠 **Skills**\n\n"
            "• Programming: Python, C, C++, Java\n"
            "• Web: HTML, CSS, JavaScript\n"
            "• Databases: MySQL, MongoDB (Basics)\n"
            "• Core Subjects: DSA, OS, CN, DBMS\n"
            "• Tools: Git, VS Code"
        )

    # Projects
    if "project" in q:
        return (
            "📂 **Projects**\n\n"
            "1️⃣ Personal Portfolio Website\n"
            "   – HTML, CSS, JavaScript\n\n"
            "2️⃣ Student Management System\n"
            "   – Python, MySQL\n\n"
            "3️⃣ Network Security Study\n"
            "   – Firewalls, IDS, Attack Analysis"
        )

    # Education
    if "education" in q or "study" in q:
        return (
            "🎓 **Education**\n\n"
            "• Degree: Bachelor of Engineering (Computer Engineering)\n"
            "• Year: Second Year\n"
            "• Focus Areas: Software, Networks, Databases, OS"
        )

    # Fallback
    return (
        "🤖 I couldn’t fully understand that.\n\n"
        "Try asking about:\n"
        "• Skills\n"
        "• Projects\n"
        "• Education\n"
        "• About Me"
    )
