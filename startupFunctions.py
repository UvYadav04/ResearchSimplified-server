from db.mongo import connect_db
from Models.model import awake_gemini, awake_client, awake_groq, awake_redis,awake_fastembed


def get_mongo(app):
    if not hasattr(app.state, "mongo_client"):
        client = connect_db()
        app.state.mongo_client = client
        app.state.research_db = client["research_db"]
    return app.state.research_db


def get_gemini(app):
    if not hasattr(app.state, "gemini"):
        gemini = awake_gemini()
        app.state.gemini = gemini
    return app.state.gemini


async def get_groq(app):
    if not hasattr(app.state, "groq"):
        groq = await awake_groq()
        app.state.groq = groq
    return app.state.groq

async def get_client(app):
    if not hasattr(app.state, "hf_client"):
        hf_client = await awake_client()
        app.state.hf_client = hf_client
    return app.state.hf_client

def get_redis(app):
    if not hasattr(app.state, "redis"):
        redis = awake_redis()
        print(redis)
        app.state.redis = redis
    return app.state.redis

def get_fastembed(app):
    if not hasattr(app.state, "fastembed"):
        fastembed = awake_fastembed()
        app.state.fastembed = fastembed
    return app.state.fastembed
