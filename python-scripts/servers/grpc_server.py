import asyncio

import embed_anything
import grpc
from embed_anything import EmbeddingModel, WhichModel

import tei_pb2 as pb2
import tei_pb2_grpc as pb2_grpc

model_cache = {}


def get_model(model_name: str):
    if model_name not in model_cache:
        print(f"Loading model: {model_name}")
        model_cache[model_name] = EmbeddingModel.from_pretrained_onnx(
            WhichModel.Bert,
            hf_model_id=model_name
        )
    return model_cache[model_name]


def embed_texts(texts, model_name):
    data = embed_anything.embed_query(texts, embedder=get_model(model_name))
    return [item.embedding for item in data]


class EmbedService(pb2_grpc.EmbedServicer):
    async def Embed(self, request, context):
        text = request.inputs
        model_name = request.model or "Qdrant/all-MiniLM-L6-v2-onnx"

        embeddings_array = embed_texts([text], model_name)

        response = pb2.EmbedResponse()
        response.embeddings.extend(embeddings_array[0])
        return response

    async def EmbedBatch(self, request, context):
        model_name = request.model or "Qdrant/all-MiniLM-L6-v2-onnx"

        embeddings_array = embed_texts(request.inputs, model_name)

        response = pb2.EmbedBatchResponse()
        for vector in embeddings_array:
            embedding_msg = pb2.Embedding()
            embedding_msg.values.extend(vector)
            response.embeddings.append(embedding_msg)

        return response


async def serve():
    server = grpc.aio.server()
    pb2_grpc.add_EmbedServicer_to_server(EmbedService(), server)
    port = 50051
    server.add_insecure_port(f"[::]:{port}")
    await server.start()
    print(f"gRPC server running on port {port}")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
