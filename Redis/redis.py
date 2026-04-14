import numpy as np
from SafeExecution.safeExecution import safeExecution
from redis.commands.search.query import Query

class Redis:
    def __init__(self, redis, cohere_client,session_id):
        self.redis = redis
        self.session = session_id
        self.cohere = cohere_client
        self.index = "vector_json_idx"

    @safeExecution
    def add_to_chat(self, messages, embeddings):
        pipe = self.redis.pipeline()

        for message, emb in zip(messages, embeddings):
            message_id = message["message_id"]
            text = message["text"]
            key = self._chunk_key(self.session, message_id)

            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            doc = {
                "item_id": message_id,
                "session_id": self.session,
                "type": "message",
                "content": text,
                "embedding": emb,
            }

            pipe.json().set(key, "$", doc)

        pipe.execute()
        return key

    def delete_session_chunks(self):
        pattern = f"{self._chunk_key(self.session, '*')}"

        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor=cursor, match=pattern, count=100)

            if keys:
                self.redis.delete(*keys)

            if cursor == 0:
                break

    @safeExecution
    def _chunk_key(self,session_id:str, chunk_id):
        return f"jdoc:chunk:{session_id}:{chunk_id}"

    @safeExecution
    def add_chunks_batch(self, chunks: list, embeddings: list):
        self.delete_session_chunks()
        pipe = self.redis.pipeline()

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]
            key = self._chunk_key(self.session, chunk_id)

            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            doc = {
                "item_id": chunk_id,
                "session_id": self.session,
                "type": "chunk",
                "content": text,
                "embedding": emb,
            }

            pipe.json().set(key, "$", doc)

        pipe.execute()

    @safeExecution
    def get_chunk_by_id(self, chunk_id: str):

        key = self._chunk_key(self.session, chunk_id)

        doc = self.redis.json().get(key)

        if not doc:
            return None

        if doc.get("session_id") != self.session:
            return None

        return doc.get("content")

    @safeExecution
    def search(self,query_original, query_embedding, top_k=5, doc_type=None):
        base_filter = f"@session_id:{{{self.session}}}"

        if doc_type:
            base_filter += f" @type:{{{doc_type}}}"

        if not isinstance(query_embedding, list):
            query_embedding = list(query_embedding)

        query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        query_str = f"({base_filter})=>[KNN {top_k} @embedding $vec AS score]"

        query = (
            Query(query_str)
            .sort_by("score")
            .paging(0, top_k)
            .return_fields("content", "type", "score")
            .dialect(2)
        )

        try:
            results = self.redis.ft("vector_json_idx").search(
                query,
                query_params={"vec": query_vector}
            )
        except Exception as e:
            print(e)
            return ""


        output = []
        for doc in results.docs:

            output.append({
                "id": doc.id,
                "content": getattr(doc, "content", ""),
                "type": getattr(doc, "type", ""),
                "score": float(getattr(doc, "score", 0)),
            })

        docs = [item["content"] for item in output]


        if len(docs) > 0:
            response = self.cohere.rerank(
                model="rerank-v4.0-pro",
                query=query_original,
                documents=docs,
                top_n=5,
            )

            indexes = [item.index for item in response.results]  # ✅ FIX


            return "\n".join([docs[index] for index in indexes])
        return "\n".join(docs)


