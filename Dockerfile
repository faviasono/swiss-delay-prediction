FROM python:3.8.5


WORKDIR /code


# Copy and install requirements
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy contents from your local to your docker container
COPY ./app /code/app
COPY ./models/model_ckpt_0.txt /code/models/model_ckpt_0.txt

# 
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8082"]
