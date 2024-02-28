# api/hello.py

from flask import Flask

app = Flask(__name__)

@app.route("/api/hello")
def app():
    return {"message": "Hello, world!"}
