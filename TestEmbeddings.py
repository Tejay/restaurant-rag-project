from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences to encode
sentences = ["What are the best Italian restaurants in London?",
             "Where can I find good Italian food in London?",
             "How's the weather today?"]

# Generate embeddings
embeddings = model.encode(sentences)

# Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print(f"Sentence: {sentence}")
    print(f"Embedding: {embedding[:10]}...")  # Print first 10 dimensions for brevity
