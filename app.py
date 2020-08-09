"""App runner"""
from app import app


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=500)
    # app.run()#debug=True)
