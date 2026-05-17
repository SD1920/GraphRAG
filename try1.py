from pyTigerGraph import TigerGraphConnection
from dotenv import load_dotenv
import os

load_dotenv()

conn = TigerGraphConnection(
    host=os.getenv("TIGERCLOUD_HOST"),
    graphname=os.getenv("TIGERCLOUD_GRAPHNAME"),
    apiToken=os.getenv("TIGERCLOUD_TOKEN")
)

print(conn.echo())