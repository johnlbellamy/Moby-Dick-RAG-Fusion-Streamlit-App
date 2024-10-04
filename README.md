# Moby Dick RAG Fusion and Streamlit App

A streamlined, RAG-fusion project featuring two Dockerized services, easily managed via Docker Compose. This project is designed to be lightweight and easy to set up, with minimal configuration required to get started.

# Overview

This repository contains two services, each packaged in a separate Docker container. Both services can be easily orchestrated using Docker Compose, making it simple to deploy and scale.

The main objective is to provide a fast and simple environment where both services can interact seamlessly. Whether you are using this project for development, testing, or deploying in production, the setup is designed to be straightforward and efficient.

# Services
Service 1: FastAPI RAG API take a post {"query": "Why does Ahab hate Moby Dick?"} 
and returns the answer based on the book. If ChatGPT doesn't know an answer it responds : "I don't know."\
Service 2: Streamlit Application.

# Prerequisites

Before getting started, make sure you have the following installed on your machine:

Docker
Docker Compose
Installation

## 1. Clone the repository
    git clone https://github.com/yourusername/project-name.git
    
    cd project-name
## 2. Build and run the services
To build and start the services, simply use Docker Compose:

    docker-compose up --build

This command will:

Build the Docker images for both services.
Start the services in the correct order as specified in the docker-compose.yml file.
Expose necessary ports for external access (if applicable).
3. Verify everything is running
Once the services are up and running, you can verify the status by using:

bash
Copy code
docker-compose ps
You should see both services up and running.

# Project Structure

<pre>
. 
├── docker-compose.yml           # Docker Compose configuration file 
├── rag_app/ 
|   └──docker 
│   |  ├── Dockerfile            # Dockerfile for RAG API 
│   |  └── app.py                # FastAPI API 
|   |  └── query.py              # Data class for API     
|   └── tools/ 
|   |   ├── tools.py             # Langchain tools 
|   └── utils
|       utils.py 
|
├── ui/                  
│   └──docker
|   |   └──Dockerfile            # Dockerfile for Streamlit 
|   └── app.py 
│   
└── README.md                    # This file 
</pre>

# Useful Commands

Here are some additional Docker Compose commands you might find helpful:

Stop all services:

    docker-compose down

Get logs for services:

    docker-compose logs service_name

Rebuild a specific service:

    docker-compose up --build service_name