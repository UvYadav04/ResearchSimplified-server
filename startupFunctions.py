from db.mongo import connect_db
from Models.model import (
    awake_gemini,
    awake_client,
    awake_groq,
    awake_redis,
    awake_fastembed,
    awake_falAi,
    awake_coherent,
)
from SafeExecution.safeExecution import safeExecution


@safeExecution
def get_mongo(app):
    if not hasattr(app.state, "mongo_client"):
        client = connect_db()
        app.state.mongo_client = client
        app.state.research_db = client["research_db"]
    return app.state.research_db


@safeExecution
def get_gemini(app):
    if not hasattr(app.state, "gemini"):
        gemini = awake_gemini()
        app.state.gemini = gemini
    return app.state.gemini


@safeExecution
async def get_groq(app):
    if not hasattr(app.state, "groq"):
        groq = await awake_groq()
        app.state.groq = groq
    return app.state.groq


@safeExecution
async def get_client(app):
    if not hasattr(app.state, "hf_client"):
        hf_client = await awake_client()
        app.state.hf_client = hf_client
    return app.state.hf_client


@safeExecution
def get_redis(app):
    if not hasattr(app.state, "redis"):
        redis = awake_redis()
        app.state.redis = redis
    return app.state.redis


@safeExecution
def get_fastembed(app):
    if not hasattr(app.state, "fastembed"):
        fastembed = awake_fastembed()
        app.state.fastembed = fastembed
    return app.state.fastembed


@safeExecution
def get_falAI(app):
    if not hasattr(app.state, "falAi"):
        falAi = awake_falAi()
        app.state.falAi = falAi
    return app.state.falAi


def get_coherent(app):
    if not hasattr(app.state, "cohere"):
        cohere = awake_coherent()
        app.state.cohere = cohere
    return app.state.cohere
