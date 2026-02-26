import chromadb

client = chromadb.PersistentClient(path="./chroma_data")
col = client.get_or_create_collection("lens_case_studies")

col.add(
    ids=["cs1"],
    documents=["Client: Gym in Colombo. Problem: low leads. Fix: landing page + speed + SEO. Result: +32% leads in 60 days."],
    metadatas=[{"industry": "fitness", "service": "seo+cro"}],
)

res = col.query(query_texts=["need more leads for a gym"], n_results=1)
print(res)