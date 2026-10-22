import argparse

import uvicorn

parser = argparse.ArgumentParser(description="Run the ISL Translator web app")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

print(f"ISL Translator: http://{'localhost' if args.host == '127.0.0.1' else args.host}:{args.port}")
uvicorn.run("server.app:app", host=args.host, port=args.port)
