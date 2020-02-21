# H1-B Visa Prediction Backend

### Running this project locally
To run locally, first you'll need to do the following:
- Install dependencies with `pip install -r requirements.txt`
  - You might need to use `pip3` instead of `pip`, depending on your python/pip installation
- Run the server with the command `FLASK_ENV=development flask run`
  - If you want to run in production mode you can just use the command `flask run` instead
  - In case of error running the command, you may need to add `FLASK_APP=server.py` to the start of it