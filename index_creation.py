# run the following commands to download the dataset:
# kaggle datasets download -d Cornell-University/arxiv
# unzip arxiv.zip

import os
from pinecone import Pinecone, ServerlessSpec, Vector
from sentence_transformers import SentenceTransformer
import json
from dotenv import load_dotenv

load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')


if __name__ == '__main__':
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

    index_name = 'arxiv-abs'
    dimension = 384
    metric = 'cosine'

    index_exists = any(index['name'] == index_name for index in pc.list_indexes())
    if not index_exists:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    index = pc.Index(index_name)

    categories = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.NE']

    vectors = []
    with open('arxiv-metadata-oai-snapshot.json', 'r') as f:
        for line in f:
            line = json.loads(line)
            if not any(cat in line['categories'] for cat in categories):
                continue
            embedding = model.encode(f"{line['title']}\n{line['abstract']}")
            vectors.append(Vector(id=line['id'], values=embedding.tolist()))
            if len(vectors) == 100:
                index.upsert(vectors=vectors)
                vectors = []

    if len(vectors) > 0:
        index.upsert(vectors=vectors)

