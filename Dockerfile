# streamlit-face-recognition-app/Dockerfile

FROM python:3.11-slim

WORKDIR /streamlit-face-recognition-app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    # git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/sergejlembke/streamlit-face-recognition-app .

COPY ./requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# ENTRYPOINT ["sh", "-c", "cd streamlit_app && streamlit run main.py --server.port=8501 --server.address=0.0.0.0"]

ENV PYTHONPATH=/streamlit-face-recognition-app

