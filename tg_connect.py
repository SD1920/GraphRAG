from pyTigerGraph import TigerGraphConnection

conn = TigerGraphConnection(
    host="https://tg-07ee6ea2-13d0-4688-afa1-c8b64a8d0b4f.tg-2635877100.i.tgcloud.io",
    graphname="RAG",
    apiToken="hfkjfsgfinsgkf49l5b9frqc1uhmhc9o",
    useCert=False
)

print(conn.echo())
print(conn.getVersion())