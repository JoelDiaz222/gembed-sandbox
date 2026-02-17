from typing import List

import embed_anything
from embed_anything import EmbeddingModel
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model_cache = {}


def get_model(model_name: str):
    if model_name not in model_cache:
        print(f"Loading model: {model_name}")
        model_cache[model_name] = EmbeddingModel.from_pretrained_hf(model_id=model_name)
    return model_cache[model_name]


def embed_texts(texts, model_name):
    data = embed_anything.embed_query(texts, embedder=get_model(model_name))
    return [item.embedding for item in data]


class EmbeddingRequest(BaseModel):
    model: str
    input: List[str]


class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str


@app.post("/v1/embed")
async def create_embeddings(request: EmbeddingRequest):
    embeddings = embed_texts(request.input, request.model)

    data = [
        EmbeddingData(embedding=emb, index=i)
        for i, emb in enumerate(embeddings)
    ]

    return EmbeddingResponse(data=data, model=request.model)


if __name__ == "__main__":
    import uvicorn

    print("Starting HTTP server on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
