from app import app


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000) #helps decide what internal address to host the website on. This will host the app at 0.0.0.0:5000
