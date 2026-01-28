# AI-Powered Process Monitoring & Recognition System

## 📌 Project Overview

This Final Year Project (FYP) is a high-performance system designed for automated process monitoring and facial analytics.

Unlike standard scripts, this project is built as a production-ready web service using **FastAPI**, capable of integrating deep learning models with a persistent **MySQL** database.

## 🚀 Key Features

* 
**Facial Detection & Landmarks:** Utilizes **RetinaFace** and **MediaPipe** for precise facial area localization and landmark tracking.


* 
**High-Accuracy Recognition:** Implements **FaceNet (PyTorch)** and **ArcFace** for identity verification and representation.


* 
**Real-time API:** Leverages **FastAPI** and **Uvicorn** for low-latency, asynchronous communication.


* 
**Database Integration:** Uses **SQLAlchemy** and **PyMySQL** to manage user data, logs, and system states.


* 
**Hybrid AI Framework:** Supports multiple backend engines including **TensorFlow (macOS optimized)** and **PyTorch**.



## 🛠 Tech Stack

* 
**Core Language:** Python.


* 
**Web Framework:** FastAPI, Jinja2 (for frontend rendering), Starlette.


* 
**AI/Deep Learning:** Torch, TensorFlow, Keras, Timm, EfficientNet.


* 
**Computer Vision:** OpenCV (Headless), RetinaFace, MediaPipe.


* 
**Data Management:** MySQL, SQLAlchemy, Pydantic.



## 📥 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Cye0/FYP-Project.git
cd FYP-Project

```

### 2. Environment Setup

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Database Configuration

Ensure you have a **MySQL** instance running and update your connection string in the configuration settings (referencing **SQLAlchemy** and **PyMySQL**).

### 5. Run the Application

```bash
uvicorn main:app --reload

```

## 📝 Usage

* Access the interactive API documentation at `http://127.0.0.1:8000/docs`.


* The system can process image streams via WebSockets or standard HTTP POST requests as defined in the FastAPI routes.
