# make_token.py
import requests

host = "https://tg-07ee6ea2-13d0-4688-afa1-c8b64a8d0b4f.tg-2635877100.i.tgcloud.io"
secret = "b00057dienelcvadii4akbm2of19p3ui"

response = requests.get(
    f"{host}/restpp/requesttoken",
    params={"secret": secret, "graph": "RAG"}
)
print(response.status_code)
print(response.text)