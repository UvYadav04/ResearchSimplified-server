from db.mongo import connect_db

def get_mongo(app):
    if not hasattr(app.state, "mongo_client"):
        client = connect_db()
        app.state.mongo_client = client
        app.state.research_db = client["research_db"]
    return app.state.research_db
