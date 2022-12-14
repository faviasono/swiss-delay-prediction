# Swiss Ariline delay-prediction

This repo contains the code for the assignment from Swiss Airline.
The goal was to develop a ML Model for detecting delay at depature.


Here's the Google Slides presentation: https://docs.google.com/presentation/d/1GzmNDbNrdxAQjwjwjQbp1fHU-4WG97ESRY3_O5Xoc20/edit?usp=sharing

n.b. Links to Weight & Biases (W&B) runs won't work becuase the project is private. 


## APIs

The model was served as Micro-service using FastAPI and Dockerized.

To try out you can either use directly `uvicorn main.app:main --port 80` or build the Docker image and start the container using:

 ```python
 
docker build -t swiss_delay_api .       
docker run -d --name swiss_delay_container -p 80:80 swiss_delay_api      
```

You can open the browser at `localhost:8080/docs` to have the specification of the API and to try it with some inputs.

## Tools

The model was created with Scikit-learn, Imbalance-learn, LightGBM, Python, Weight & Biases, FastAPI, and Docker 🚀
