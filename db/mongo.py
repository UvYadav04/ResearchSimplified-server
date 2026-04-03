from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
load_dotenv()
import certifi
import os

uri = os.environ.get("MONGO_URI")


def connect_db():
    client = MongoClient(
        uri,
        server_api=ServerApi("1"),
        tls=True,
        tlsCAFile=certifi.where(),
        uuidRepresentation="standard",
    )
    try:
        client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client
    except Exception as e:
        print(e)
