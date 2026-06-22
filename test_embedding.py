from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embedding = model.encode(
    "What is LangChain?"
)

print(type(embedding))
print(len(embedding))
