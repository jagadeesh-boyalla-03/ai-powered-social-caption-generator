# AI-Powered Multi-Platform Social Media Caption Generator & Studio

[![FastAPI](https://img.shields.io/badge/FastAPI-0.139+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Google Gemini](https://img.shields.io/badge/Google%20Gemini-Multimodal%20Vision-4285F4.svg?style=flat&logo=google)](https://ai.google.dev)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat&logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-26%20Passed-brightgreen.svg?style=flat&logo=pytest)](file:///c:/Users/boyal/OneDrive/Desktop/ai-projects/1/tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent, full-stack social media studio that transforms visual images and videos into platform-optimized, high-engagement captions and hashtags with multimodal AI powered by **Google Gemini**.

---

## 🌟 Key Features

* **Multimodal Vision AI**: Analyzes uploaded photos and visual assets using Google Gemini to capture visual context, vibe, mood, and subject details.
* **Multi-Platform Tailoring**: Automatically generates distinct, formatted captions adhering to platform rules, character limits, tone, and hashtag strategies for:
  * 📸 **Instagram** (up to 2,200 chars, storytelling, trending & niche tags)
  * 🐦 **X / Twitter** (strictly ≤ 280 chars, punchy hooks, targeted tags)
  * 👥 **Facebook** (conversational, community questions, discussion hooks)
  * 💼 **LinkedIn** (professional, thought leadership, business insights)
  * 📌 **Pinterest** (search-optimized, aesthetic descriptions)
  * ▶️ **YouTube** (description formatting, timestamps, subscribe CTAs)
  * 🟢 **Kick** (high-energy stream titles, hype hooks)
* **User Authentication**: Secure user registration, sign-in, session tokens, and PBKDF2 password hashing with unique salts.
* **Caption History & Management**: Registered users automatically save generation history to an isolated SQLite store with options to inspect, copy, or delete items.
* **Guest Mode**: 1-click generation without mandatory login for rapid previews.
* **Interactive Web App**: Sleek, modern single-page dashboard with real-time copy-to-clipboard, tone customization, and platform toggles.
* **Tested & Reliable**: Full automated test suite covering authentication, API endpoints, and caption generation with 100% passing tests.

---

## 📁 Project Structure

```text
ai-powered-social-caption-generator/
│
├── api/
│   └── index.py            # FastAPI backend, Gemini vision engine, Auth & DB routes
│
├── frontend/
│   └── index.html          # Modern Single Page Application (SPA) web client
│
├── static/
│   └── auth_robot.jpg      # Static illustration assets
│
├── data/
│   └── auth.db             # SQLite database (Users, Sessions, Caption History)
│
├── tests/
│   ├── conftest.py         # Pytest configuration, fixtures & test DB setup
│   ├── test_api.py         # Health, SPA routing & history endpoint tests
│   ├── test_auth.py        # PBKDF2 password hashing & session token tests
│   └── test_captions.py    # Gemini retry, fallback & platform clamping tests
│
├── origin/                 # Standalone prototypes, scripts & deployment helpers
│   ├── 1_caption_image.py
│   ├── 6_social_media_captions.py     # Standalone terminal caption generator
│   ├── 7_streamlit_social_demo.py     # Streamlit interactive UI demo
│   └── deploy.py                      # Git auto-commit & Render deploy trigger
│
├── requirements.txt        # Python dependency manifest
├── vercel.json             # Vercel serverless deployment config
└── README.md               # Project documentation
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Jagadeesh-31/ai-powered-social-caption-generator.git
cd ai-powered-social-caption-generator
```

### 2. Set Up a Virtual Environment

* **Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```

* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Your API Key

Create a `.env` file in the project root directory:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

*(You can get a free API key at [Google AI Studio](https://aistudio.google.com/)).*

---

## 💻 Running the Application

### Option A: Modern Web Application (FastAPI + Web UI)

Start the Uvicorn development server:

```powershell
uvicorn api.index:app --reload --port 8000
```

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

* Interactive Swagger API docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**
* Redoc documentation: **[http://localhost:8000/redoc](http://localhost:8000/redoc)**

---

### Option B: Streamlit Demo

To run the interactive Streamlit UI prototype:

```powershell
streamlit run origin/7_streamlit_social_demo.py
```

---

### Option C: Standalone CLI Script

Generate captions directly for any image file via the command line:

```powershell
python origin/6_social_media_captions.py path/to/image.jpg --platforms instagram,x,linkedin --tone playful
```

---

## 🧪 Running Automated Tests

Run the full Pytest test suite with:

```powershell
pytest -v
```

All 26 tests cover:
- Authentication & Password Security (PBKDF2 salt uniqueness & validation)
- Token-based Session Management & Expiration
- Caption Length Clamping & Platform Formatting Rules
- Fallback Generators & Gemini Error Handling
- REST API Endpoints, Healthchecks & SPA Delivery

---

## 🔌 API Endpoints Summary

| Method | Endpoint | Description | Auth Required |
|---|---|---|:---:|
| `GET` | `/health` | Healthcheck and server status | No |
| `POST` | `/api/auth/register` | Register new user account | No |
| `POST` | `/api/auth/login` | Authenticate user and issue session token | No |
| `GET` | `/api/auth/me` | Fetch active user profile | Yes (Bearer Token) |
| `POST` | `/api/auth/logout` | Invalidate current session | Yes (Bearer Token) |
| `POST` | `/api/generate` | Generate captions from uploaded image/video | Optional (Guest/User) |
| `GET` | `/api/history` | Retrieve user's previous generated captions | Yes (Bearer Token) |
| `DELETE` | `/api/history/{id}` | Delete a specific saved caption item | Yes (Bearer Token) |

---

## 🚢 Deployment

### Vercel Deployment
The repository includes a ready-to-deploy [`vercel.json`](file:///c:/Users/boyal/OneDrive/Desktop/ai-projects/1/vercel.json) configuration:
```bash
vercel
```

### Render Deployment
Use the included automated deployment script to stage, commit, push, and trigger your Render deploy hook:
```powershell
python origin/deploy.py "Update feature XYZ"
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
