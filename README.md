# Concrete Defect Detection 🏗️

<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=2E9EF7&center=true&vCenter=true&width=520&lines=Concrete+Defect+Detection;Enterprise+Structural+Health+Monitoring;BIM+Ready+%26+Automated+Reporting;Built+with+FastAPI+%2B+React" alt="Typing SVG" />
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

---

## 🚀 About The Project

**Concrete Defect Detection** is an enterprise-grade platform designed to automate the inspection of civil infrastructure. It leverages advanced Computer Vision to identify, classify, and assess the severity of structural defects in concrete columns.


Designed for **Civil Engineers** and **Inspectors**, it transforms manual, error-prone visual inspections into a digital, auditable, and quantifiable workflow.

## 📸 Screenshots

<div align="center">
  <img src="screenshots/1.png" width="45%" />
  <img src="screenshots/2.png" width="45%" />
</div>
<div align="center">
  <img src="screenshots/3.png" width="45%" />
  <img src="screenshots/4.png" width="45%" />
</div>

---

## 💡 Key Features

<div align="center">

| Feature | Description |
| :--- | :--- |
| 🕵️‍♂️ **Multi-Class Detection** | Detects 5 damage types: **Crack**, **Spalling**, **Honeycomb**, **Segregation**, **Corrosion**. |
| 📏 **Severity Assessment** | Auto-classifies damage as **Minor**, **Moderate**, or **Severe** based on visual features. |
| 🛠️ **Smart Recommendations** | Provides ISO/ACI-standard repair solutions based on damage type and severity. |
| 🏗️ **BIM Integration** | Exports "IFC Overlay" JSON data for direct integration with **Autodesk Revit** / Dynamo. |
| 📊 **History & Analytics** | Tracks defect progression over time with a responsive dashboard and filtering. |

</div>

---

## 🛠️ Tech Stack

### 🧠 AI & Core
*   **Model**: YOLO / Custom CNN optimized via **ONNX Runtime** for high-performance CPU inference.
*   **Processing**: OpenCV, NumPy, Pillow for pre/post-processing.

### 🔌 Backend (API)
*   **Framework**: **FastAPI** (Python) for high-performance async endpoints.
*   **Storage**: JSON-based storage (scales to SQLite/PostgreSQL easily).

### 💻 Frontend (UI)
*   **Framework**: **React** (Vite) + **TypeScript**.
*   **Styling**: **TailwindCSS** + **Panty** (Framer Motion) for modern, responsive cards.

---

## ⚙️ Installation & Setup

This is a **Monorepo** containing both the AI Backend and React Frontend.

### Prerequisites
*   Python 3.9+
*   Node.js 18+

### 1️⃣ Backend Setup
```bash
cd src/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Server
python app.py
# 📡 Server runs at http://localhost:8000
```

### 2️⃣ Frontend Setup
```bash
cd src/frontend

# Install dependencies
npm install

# Run Dev Server
npm run dev
# 💻 UI runs at http://localhost:5173
```

---

## 📂 Project Structure

A professional ML engineering structure separating research, artifacts, and production code.

```text
App-Crack/
├── data/                   # Data storage (Raw/Processed) - Git Ignored
├── models/                 # Model Artifacts (.onnx, .pt)
│   ├── production/         # Active production models
│   └── archive/            # Versioned backups
├── notebooks/              # Jupyter Notebooks for R&D
├── src/                    # Production Source Code
│   ├── backend/            # FastAPI Application
│   └── frontend/           # React Application
└── scripts/                # Utility scripts
```

---

<div align="center">

⚡ *Automating Infrastructure Safety, One Crack at a Time.*

</div>
