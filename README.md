# AI-Powered Multi-Platform Social Media Caption Generator

This project translates visual image and video context into highly engaging, platform-specific social media captions (for Instagram, X/Twitter, Facebook, LinkedIn, Pinterest, YouTube, and Kick) with ultra-fast inference powered by **Groq AI**.

---

## Local Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Jagadeesh-31/ai-powered-social-caption-generator.git
   cd ai-powered-social-caption-generator
   ```

2. **Initialize Virtual Environment:**
   * **Windows:**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   * **Mac/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Your API Key:**
   Create a `.env` file or set the environment variable:
   * **In `.env` file:**
     ```env
     GEMINI_API_KEY=your-gemini-key
     ```
   * **Windows (PowerShell):** `$env:GEMINI_API_KEY="your-gemini-key"`
   * **Mac/Linux:** `export GEMINI_API_KEY="your-gemini-key"`

5. **Run the Web Application (FastAPI + Modern Web UI):**
   ```bash
   uvicorn api.index:app --reload --port 8000
   ```
   Open `http://localhost:8000` in your browser.

---

## CLI Usage

Generate captions from any image via terminal:
```bash
python 6_social_media_captions.py test.jpg --platforms instagram,x,linkedin --tone playful
```
