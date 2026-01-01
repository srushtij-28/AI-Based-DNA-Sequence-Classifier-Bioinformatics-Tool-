from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

dna_data = [
    "ATCGGCTAAGTC",   # healthy
    "ATCGGCTAACGT",
    "ATCGGCTAAGTA",
    "ATCGGCTTTGAA",   # mutated
    "ATCGGCTGGGAA",
    "ATCGGCTTTAAA"
]

labels = [0,0,0,1,1,1]   # 0 = Healthy, 1 = Mutated

vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3))
X = vectorizer.fit_transform(dna_data)

model = MultinomialNB()
model.fit(X, labels)

def predict(seq):
    vec = vectorizer.transform([seq])
    result = model.predict(vec)[0]
    return "Mutated 🦠" if result == 1 else "Healthy 🧬"

print("🧬 DNA Sequence Classifier\n")
dna = input("Enter DNA sequence: ")
print("\nResult:", predict(dna))
