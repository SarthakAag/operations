from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Past issues + fixes
data = [
    ("high cpu usage", "Restart Service"),
    ("database error", "Check DB Connection"),
    ("memory leak", "Clear Cache"),
    ("timeout error", "Increase Timeout")
]

texts = [d[0] for d in data]
solutions = [d[1] for d in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

def recommend_solution(new_issue):
    new_vec = vectorizer.transform([new_issue])
    similarity = cosine_similarity(new_vec, X)

    index = similarity.argmax()
    return solutions[index]