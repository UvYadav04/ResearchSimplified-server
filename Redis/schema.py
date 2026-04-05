# redis/schema.py

from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType

INDEX_NAME = "vector_json_idx"


def create_index(redis_client):
    try:
        redis_client.ft(INDEX_NAME).info()
        print("Index already exists")
        return
    except:
        print("Creating index...")

    schema = (
        TextField("$.content", as_name="content"),
        TagField("$.item_id", as_name="item_id"),
        TagField("$.session_id", as_name="session_id"),
        TagField("$.type", as_name="type"),
        VectorField(
            "$.embedding",
            "HNSW",
            {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"},
            as_name="embedding",
        ),
    )

    definition = IndexDefinition(prefix=["jdoc:"], index_type=IndexType.JSON)

    redis_client.ft(INDEX_NAME).create_index(schema, definition=definition)
    print("Index created 🚀")
