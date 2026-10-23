from waitress import serve
from .web import app

if __name__ == "__main__":
    print("InspectAI: http://127.0.0.1:8100", flush=True)
    serve(app, host="127.0.0.1", port=8100, max_request_body_size=8 * 1024 * 1024)
