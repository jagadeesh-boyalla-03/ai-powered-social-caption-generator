"""
====================================================================================================
PROJECT: AI-Powered Multi-Platform Social Media Caption Generator & Studio
BACKEND: FastAPI + Google Gemini Multimodal Vision AI + SQLite Authentication & History System
====================================================================================================
This module powers the complete backend application workflow:
  - STEP 1: Environment initialization, logging & CORS configuration.
  - STEP 2: SQLite database initialization (Users, Sessions, Caption History).
  - STEP 3: Cryptographic password hashing (PBKDF2) & session token management.
  - STEP 4: Authentication endpoints (Register, Login, Profile Me, Logout).
  - STEP 5: History retrieval and deletion endpoints.
  - STEP 6: Multi-platform social caption formatting rules and prompt engineering.
  - STEP 7: Multimodal image/video analysis and AI inference with Google Gemini & fallbacks.
  - STEP 8: Static asset and Single Page Application (SPA) routing.
====================================================================================================
"""

import os
import io
import re
import json
import logging
import sqlite3
import hashlib
import secrets
import tempfile
import shutil
import random
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from PIL import Image
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --------------------------------------------------------------------------------------------------
# STEP 1: LOGGING & ENVIRONMENT CONFIGURATION & FASTAPI APPLICATION SETUP
# --------------------------------------------------------------------------------------------------
# Configure structured logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("social_caption_api")

# Load API keys and settings from the .env configuration file
load_dotenv()
logger.info("Environment variables loaded. Log level set to %s", LOG_LEVEL)

# Initialize the FastAPI server instance
app = FastAPI(
    title="AI Social Media Caption API with Authentication",
    description="Transforms visual media into platform-optimized social media captions with user auth & history.",
    version="2.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS) to allow browser client requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------------------
# STEP 2: DATABASE SETUP (SQLite FOR USERS, SESSIONS & CAPTION HISTORY)
# --------------------------------------------------------------------------------------------------
def get_db_path() -> str:
    """Returns the configured database path or default data/auth.db."""
    db_path = os.environ.get("DATABASE_PATH")
    if db_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
            return db_path
        except OSError:
            pass

    # Check for serverless read-only environments (Vercel, AWS Lambda, etc.)
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return os.path.join(tempfile.gettempdir(), "auth.db")

    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    try:
        os.makedirs(default_dir, exist_ok=True)
        return os.path.join(default_dir, "auth.db")
    except OSError:
        # Fallback to system temp directory if local workspace is read-only
        return os.path.join(tempfile.gettempdir(), "auth.db")

def get_db() -> sqlite3.Connection:
    """Helper function to create a new database connection with dictionary-like row access."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite database tables for user management, sessions, and history logging."""
    try:
        db_path = get_db_path()
        logger.info("Initializing database tables at path: %s", db_path)
        with get_db() as conn:
            cursor = conn.cursor()
            
            # 1. Users Table: Stores registered user credentials and registration date
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            # 2. Sessions Table: Stores active bearer authentication tokens with expiration dates
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            
            # 3. History Table: Stores generated caption records linked to user accounts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    tone TEXT NOT NULL,
                    captions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """)
            conn.commit()
        logger.info("Database schema initialized and verified successfully.")
    except Exception as e:
        logger.error("Database schema initialization non-fatal error: %s", e)

# Execute table creation on startup with error safeguard
try:
    init_db()
except Exception as e:
    logger.error("Failed to execute init_db on startup: %s", e)

# --------------------------------------------------------------------------------------------------
# STEP 3: CRYPTOGRAPHIC PASSWORD HASHING & TOKEN HELPERS
# --------------------------------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hashes a plaintext password using PBKDF2-HMAC-SHA256 with a unique random 16-byte salt."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 100000)
    return f"{salt}:{key.hex()}"

def verify_password(password: str, stored_hash: str) -> bool:
    """Safely verifies a candidate password against the stored salt and PBKDF2 hash."""
    try:
        salt, key_hex = stored_hash.split(":", 1)
        test_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 100000)
        return secrets.compare_digest(test_key.hex(), key_hex)
    except Exception as e:
        logger.warning("Password verification error: %s", e)
        return False

def create_session(user_id: int) -> str:
    """Generates a secure 32-byte session token valid for 30 days and stores it in the database."""
    token = secrets.token_urlsafe(32)
    created_at = datetime.utcnow().isoformat()
    expires_at = (datetime.utcnow() + timedelta(days=30)).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, created_at, expires_at)
        )
        conn.commit()
    logger.debug("Created session token for user_id=%d (expires: %s)", user_id, expires_at)
    return token

def get_user_from_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Validates an incoming bearer token and returns the authenticated user data if valid."""
    if not token:
        return None
    token_clean = token.replace("Bearer ", "").strip()
    if not token_clean:
        return None
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.id, u.name, u.email, u.created_at, s.expires_at 
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
        """, (token_clean,))
        row = cursor.fetchone()
        if row:
            # Check if session has expired
            if datetime.fromisoformat(row["expires_at"]) > datetime.utcnow():
                return {
                    "id": row["id"],
                    "name": row["name"],
                    "email": row["email"],
                    "created_at": row["created_at"]
                }
            else:
                logger.info("Session expired for user_id=%d. Removing expired token.", row["id"])
                # Remove expired session
                conn.execute("DELETE FROM sessions WHERE token = ?", (token_clean,))
                conn.commit()
    return None

# --------------------------------------------------------------------------------------------------
# STEP 4: DATA MODELS & PLATFORM STYLE RULES
# --------------------------------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

# Platform character limits and copywriting tone requirements
PLATFORM_RULES = {
    "instagram": {
        "label": "Instagram",
        "max_chars": 2200,
        "style": "warm, visual, engaging; 5-8 trending and niche hashtags; 1-2 emojis; engaging call-to-action",
    },
    "x": {
        "label": "X (Twitter)",
        "max_chars": 280,
        "style": "punchy, witty, high-impact concise text; 2-3 targeted hashtags; strictly under 280 characters",
    },
    "facebook": {
        "label": "Facebook",
        "max_chars": 600,
        "style": "conversational, community-focused, engaging question; 4-5 relevant hashtags",
    },
    "linkedin": {
        "label": "LinkedIn",
        "max_chars": 750,
        "style": "professional, insight-driven, practical career/business takeaways; 5+ industry hashtags",
    },
    "pinterest": {
        "label": "Pinterest",
        "max_chars": 500,
        "style": "keyword-rich, aesthetic and inspirational; 5-7 search hashtags",
    },
    "youtube": {
        "label": "YouTube",
        "max_chars": 5000,
        "style": "video description format with key takeaways, subscribe CTA, and 5-10 tags/hashtags",
    },
    "kick": {
        "label": "Kick",
        "max_chars": 150,
        "style": "high energy stream title, hype hook, 2-3 hashtags",
    },
}

class SocialCaptions(BaseModel):
    """Pydantic schema for structured JSON output from Gemini."""
    instagram: Optional[str] = None
    x: Optional[str] = None
    facebook: Optional[str] = None
    linkedin: Optional[str] = None
    pinterest: Optional[str] = None
    youtube: Optional[str] = None
    kick: Optional[str] = None

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_MODEL_FALLBACKS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
]

def looks_like_gemini_api_key(key: str) -> bool:
    """Validates the basic syntax and length of a Gemini API key."""
    if not key:
        return False
    return len(key.strip()) >= 20

def clamp_caption(text: str, max_chars: int) -> str:
    """Ensures the generated caption strictly respects each platform's character limit."""
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."

def build_fallback_caption(platform_key: str, description: str, tone: str, context: str) -> str:
    """Provides high-quality emergency templates in case of API rate-limiting or network issues."""
    details = context.strip() or description.strip() or "a great moment"
    templates = {
        "instagram": f"Putting in the work and enjoying the journey. {details}\n\nDouble tap if you resonate with this! ✨\n\n#Growth #Creativity #DailyInspiration #Moments #VisualStory",
        "x": f"Good things start with focus and showing up. {details} ⚡ #Focus #Momentum #Growth",
        "facebook": f"A moment worth sharing: {details}.\n\nWhat has inspired your work this week? Let us know below!\n\n#Community #Inspiration #WorkInProgress #DailyLife #Storytelling",
        "linkedin": f"Success is built through focus, consistency, and continuous improvement.\n\nKey Takeaway: {details}\n\nHow does your team cultivate this mindset?\n\n#ProfessionalGrowth #WorkEthic #Leadership #PersonalBranding #Productivity",
        "pinterest": f"Inspiration: {details}.\nA clean, confident visual focused on growth, lifestyle, and purpose.\n\n#Inspiration #Style #Design #CreativeIdeas #Aesthetic",
        "youtube": f"Today's focus: {details}\n\nA simple reminder that consistency and passion lead to results.\n\n🔔 Subscribe for more updates!\n\n#Growth #Creativity #Inspiration #VideoContent #LearnAndGrow",
        "kick": f"Locked in and live! {details} 🎮 #Stream #Gaming #LiveNow",
    }
    rules = PLATFORM_RULES.get(platform_key, {"max_chars": 500})
    return clamp_caption(templates.get(platform_key, details), rules["max_chars"])

def generate_gemini_content_with_retry(client: genai.Client, contents, max_retries: int = 3, config=None):
    """Executes Gemini API requests with automatic exponential backoff retry and model fallbacks."""
    last_exception = None
    for model_name in GEMINI_MODEL_FALLBACKS:
        for attempt in range(max_retries):
            try:
                logger.debug("Calling Gemini API: model=%s, attempt=%d/%d", model_name, attempt + 1, max_retries)
                kwargs = {"model": model_name, "contents": contents}
                if config:
                    kwargs["config"] = config
                return client.models.generate_content(**kwargs)
            except Exception as e:
                last_exception = e
                err_str = str(e).upper()
                is_transient = "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "BUSY" in err_str
                if is_transient and attempt < max_retries - 1:
                    wait_time = (1.5 ** attempt) + random.uniform(0.5, 1.0)
                    logger.warning("Transient Gemini API error (%s) on model=%s, attempt %d. Retrying in %.2fs...", e, model_name, attempt + 1, wait_time)
                    time.sleep(wait_time)
                else:
                    logger.warning("Gemini attempt failed on model=%s (attempt %d): %s", model_name, attempt + 1, e)
                    break
    if last_exception:
        logger.error("All Gemini model fallback attempts failed: %s", last_exception)
        raise last_exception
    raise Exception("Gemini service unavailable after attempting fallback models.")

# --------------------------------------------------------------------------------------------------
# STEP 5: AUTHENTICATION API ENDPOINTS
# --------------------------------------------------------------------------------------------------

@app.post("/api/auth/register")
def register_user(req: RegisterRequest):
    """Creates a new user account, hashes their password, and creates an initial login session."""
    name = req.name.strip()
    email = req.email.strip().lower()
    password = req.password

    logger.info("Processing user registration attempt for email=%s", email)

    # Input validation
    if not name or len(name) < 2:
        logger.warning("Registration rejected: name '%s' too short", name)
        raise HTTPException(status_code=400, detail="Please enter a valid full name (minimum 2 characters).")
    if not email or "@" not in email or "." not in email:
        logger.warning("Registration rejected: invalid email format '%s'", email)
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    if not password or len(password) < 6:
        logger.warning("Registration rejected: password for '%s' too short", email)
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")

    pw_hash = hash_password(password)
    created_at = datetime.utcnow().isoformat()

    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (name, email, pw_hash, created_at)
            )
            user_id = cursor.lastrowid
            conn.commit()
    except sqlite3.IntegrityError:
        logger.warning("Registration rejected: email '%s' already registered", email)
        raise HTTPException(status_code=400, detail="An account with this email address already exists. Please log in.")

    token = create_session(user_id)
    logger.info("User registered successfully: user_id=%d, email=%s", user_id, email)
    return {
        "success": True,
        "token": token,
        "user": {
            "id": user_id,
            "name": name,
            "email": email,
            "created_at": created_at
        },
        "message": f"Welcome to AI Caption Generator, {name}!"
    }

@app.post("/api/auth/login")
def login_user(req: LoginRequest):
    """Authenticates user credentials and returns a session token."""
    email = req.email.strip().lower()
    password = req.password

    logger.info("Processing login attempt for email=%s", email)

    if not email or not password:
        logger.warning("Login rejected: missing email or password")
        raise HTTPException(status_code=400, detail="Email and password are required.")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, password_hash, created_at FROM users WHERE email = ?", (email,))
        user_row = cursor.fetchone()

    if not user_row or not verify_password(password, user_row["password_hash"]):
        logger.warning("Login failed: invalid credentials for email=%s", email)
        raise HTTPException(status_code=401, detail="Invalid email or password. Please try again.")

    token = create_session(user_row["id"])
    logger.info("User logged in successfully: user_id=%d, email=%s", user_row["id"], email)
    return {
        "success": True,
        "token": token,
        "user": {
            "id": user_row["id"],
            "name": user_row["name"],
            "email": user_row["email"],
            "created_at": user_row["created_at"]
        },
        "message": f"Welcome back, {user_row['name']}!"
    }

@app.get("/api/auth/me")
def get_current_user_profile(authorization: Optional[str] = Header(None)):
    """Retrieves current user's profile and lifetime caption stats."""
    user = get_user_from_token(authorization)
    if not user:
        logger.debug("Unauthorized profile request with header: %s", authorization)
        raise HTTPException(status_code=401, detail="Session expired or invalid. Please sign in.")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as caption_count FROM history WHERE user_id = ?", (user["id"],))
        stats = cursor.fetchone()
        caption_count = stats["caption_count"] if stats else 0

    logger.debug("Profile retrieved for user_id=%d (captions_count=%d)", user["id"], caption_count)
    return {
        "authenticated": True,
        "user": user,
        "stats": {
            "total_captions_generated": caption_count
        }
    }

@app.post("/api/auth/logout")
def logout_user(authorization: Optional[str] = Header(None)):
    """Logs out user by invalidating their active database session."""
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        with get_db() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
        logger.info("User session invalidated upon logout.")
    return {"success": True, "message": "Logged out successfully."}

# --------------------------------------------------------------------------------------------------
# STEP 6: HISTORY MANAGEMENT ENDPOINTS
# --------------------------------------------------------------------------------------------------

@app.get("/api/history")
def get_user_history(authorization: Optional[str] = Header(None)):
    """Fetches up to 50 previous caption generation records for the authenticated user."""
    user = get_user_from_token(authorization)
    if not user:
        logger.warning("Unauthorized access attempt to /api/history")
        raise HTTPException(status_code=401, detail="Please sign in to view your history.")

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, description, media_type, tone, captions_json, created_at 
            FROM history 
            WHERE user_id = ? 
            ORDER BY id DESC LIMIT 50
        """, (user["id"],))
        rows = cursor.fetchall()

    history_list = []
    for r in rows:
        try:
            captions_data = json.loads(r["captions_json"])
        except Exception:
            captions_data = {}
        history_list.append({
            "id": r["id"],
            "description": r["description"],
            "media_type": r["media_type"],
            "tone": r["tone"],
            "captions": captions_data,
            "created_at": r["created_at"]
        })

    logger.info("Fetched %d history items for user_id=%d", len(history_list), user["id"])
    return {"history": history_list}

@app.delete("/api/history/{history_id}")
def delete_history_item(history_id: int, authorization: Optional[str] = Header(None)):
    """Deletes a specific history record belonging to the authenticated user."""
    user = get_user_from_token(authorization)
    if not user:
        logger.warning("Unauthorized history deletion attempt for history_id=%d", history_id)
        raise HTTPException(status_code=401, detail="Unauthorized.")

    with get_db() as conn:
        conn.execute("DELETE FROM history WHERE id = ? AND user_id = ?", (history_id, user["id"]))
        conn.commit()

    logger.info("Deleted history record id=%d for user_id=%d", history_id, user["id"])
    return {"success": True, "message": "History item removed."}

# --------------------------------------------------------------------------------------------------
# STEP 7: MULTIMODAL VISION CAPTION GENERATION
# --------------------------------------------------------------------------------------------------

@app.post("/api/generate")
@app.post("/{full_path:path}")
@app.post("/")
async def generate_captions(
    file: UploadFile = File(...),
    platforms: str = Form(...),
    tone: str = Form("Friendly and upbeat"),
    context: str = Form(""),
    api_key: str = Form(""),
    authorization: Optional[str] = Header(None),
    user_token: Optional[str] = Form("")
):
    """
    Main Multimodal AI Pipeline:
    1. Validates API key and media upload.
    2. Uses Gemini Vision model to extract semantic scene descriptions.
    3. Prompts Gemini LLM with copywriting guidelines to produce structured JSON captions.
    4. Automatically records output to user's history if authenticated.
    """
    temp_file_path = None
    uploaded_file = None
    client = None
    try:
        # Step 7.1: Resolve Gemini API Key (User-provided or server environment)
        key = api_key.strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not key:
            logger.warning("Caption generation rejected: Gemini API key missing")
            raise HTTPException(
                status_code=400,
                detail="Gemini API key is missing. Add GEMINI_API_KEY to your .env file or server environment."
            )
        if not looks_like_gemini_api_key(key):
            logger.warning("Caption generation rejected: Invalid API key format")
            raise HTTPException(
                status_code=400,
                detail="Invalid API key format. Please check your Gemini API key."
            )

        client = genai.Client(api_key=key)

        # Step 7.2: Process input media (Image or Video)
        is_video = file.content_type and file.content_type.startswith("video/")
        media_type = "video" if is_video else "image"
        logger.info("Processing media for caption generation: filename=%s, media_type=%s, tone='%s'", file.filename, media_type, tone)

        if is_video:
            file_extension = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name

            logger.info("Uploading temporary video to Gemini File API: %s", temp_file_path)
            uploaded_file = client.files.upload(file=temp_file_path)
            max_polls = 45
            poll_count = 0
            while uploaded_file.state.name == "PROCESSING" and poll_count < max_polls:
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
                poll_count += 1

            if uploaded_file.state.name == "FAILED":
                logger.error("Gemini video processing state returned FAILED")
                raise Exception("Video processing failed on Gemini.")
            
            logger.info("Video processed successfully. Requesting scene description from Gemini...")
            desc_resp = generate_gemini_content_with_retry(
                client=client,
                contents=[uploaded_file, "Provide a plain, factual, one-sentence description of this video."]
            )
            image_description = desc_resp.text.strip()
        else:
            image_bytes = await file.read()
            raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            logger.info("Image loaded successfully (%dx%d). Requesting vision description from Gemini...", raw_img.width, raw_img.height)
            desc_resp = generate_gemini_content_with_retry(
                client=client,
                contents=[raw_img, "Provide a plain, factual, one-sentence description of this image."]
            )
            image_description = desc_resp.text.strip()

        logger.info("Vision description obtained: '%s'", image_description)

        # Step 7.3: Build Platform Prompt with Copywriting Rules
        platform_list = [p.strip().lower() for p in platforms.split(",") if p.strip()]
        rules_text_list = []
        for platform_key in platform_list:
            if platform_key in PLATFORM_RULES:
                rules = PLATFORM_RULES[platform_key]
                rules_text_list.append(f"- {platform_key} ({rules['label']}): Style: {rules['style']}, Max characters: {rules['max_chars']}")
        rules_text = "\n".join(rules_text_list)

        context_line = f"\nAdditional context from user: {context}\n" if context else ""

        prompt = f"""You are an elite social media copywriter and growth strategist.
Media Description: "{image_description}"
{context_line}
Platforms & Style Requirements:
{rules_text}

MANDATORY RULES:
1. Hashtags: Include AT LEAST 5+ relevant, high-traffic hashtags for LinkedIn, Instagram, Facebook, and Pinterest (and 2-3 for X). Never skip hashtags!
2. Concise & High-Impact (No Wasted Space): Write short, punchy paragraphs with substance. Avoid excessive blank lines or fluffy filler text. Every line must deliver value.
3. Structure: 
   - Strong hook line
   - Concise body / takeaway (1-2 short tight paragraphs)
   - Quick Call to Action (CTA)
   - 5+ Hashtags on the final line(s)
4. Tone: Strongly embody the "{tone}" tone across all platforms.
5. Character Limits: The complete text + all hashtags MUST stay strictly within each platform's character limit.
6. Output: Output MUST be a JSON object containing the caption for each requested platform ID: {", ".join(platform_list)}.
"""

        # Step 7.4: Request Structured Output from Gemini
        caption_text_map = {}
        fallback_used = False
        try:
            logger.info("Requesting structured caption generation from Gemini for platforms: %s", platform_list)
            response = generate_gemini_content_with_retry(
                client=client,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SocialCaptions,
                )
            )

            if response.parsed:
                if hasattr(response.parsed, "model_dump"):
                    caption_text_map = response.parsed.model_dump()
                else:
                    caption_text_map = response.parsed.__dict__
            else:
                caption_text_map = json.loads(response.text)
        except Exception as caption_error:
            logger.warning("Error generating structured captions via Gemini: %s. Using high-quality fallback templates.", caption_error)
            fallback_used = True
            caption_text_map = {}

        # Step 7.5: Post-process captions and clamp character lengths
        results = {}
        for platform_key in platform_list:
            if platform_key not in PLATFORM_RULES:
                continue
            rules = PLATFORM_RULES[platform_key]
            caption_text = caption_text_map.get(platform_key) or ""
            if not isinstance(caption_text, str):
                caption_text = str(caption_text)

            if not caption_text.strip():
                fallback_used = True
                caption_text = build_fallback_caption(platform_key, image_description, tone, context)

            results[platform_key] = {
                "label": rules["label"],
                "caption": clamp_caption(caption_text.strip(), rules["max_chars"]),
                "max_chars": rules["max_chars"]
            }

        # Step 7.6: Automatically save generation to user history if logged in
        current_token = authorization or user_token
        current_user = get_user_from_token(current_token)
        saved_to_history = False
        if current_user:
            try:
                with get_db() as conn:
                    conn.execute(
                        """
                        INSERT INTO history (user_id, description, media_type, tone, captions_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            current_user["id"],
                            image_description,
                            media_type,
                            tone,
                            json.dumps(results),
                            datetime.utcnow().isoformat()
                        )
                    )
                    conn.commit()
                    saved_to_history = True
                logger.info("Caption generation successfully recorded in history for user_id=%d", current_user["id"])
            except Exception as history_err:
                logger.error("Failed to record history for user_id=%d: %s", current_user["id"], history_err)

        logger.info("Caption generation complete. Platforms: %s, Fallback used: %s", list(results.keys()), fallback_used)
        return {
            "description": image_description,
            "captions": results,
            "fallback_used": fallback_used,
            "saved_to_history": saved_to_history,
            "user": current_user["name"] if current_user else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error occurred in generate_captions pipeline: %s", e, exc_info=True)

        err_msg = str(e)
        status_code = 400
        if "api_key" in err_msg.lower() or "403" in err_msg or "unauthorized" in err_msg.lower():
            err_msg = "Invalid Gemini API Key. Please verify your API key."
            status_code = 401
        elif "quota" in err_msg.lower() or "rate limit" in err_msg.lower() or "429" in err_msg:
            err_msg = "Gemini API rate limit reached. Please wait a moment and try again."
            status_code = 429

        return JSONResponse(status_code=status_code, content={"detail": err_msg})

    finally:
        # Step 7.7: Resource cleanup (delete temp files and remote uploaded assets)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug("Temporary file cleaned up: %s", temp_file_path)
            except Exception as clean_err:
                logger.warning("Failed to clean up temp file %s: %s", temp_file_path, clean_err)
        if uploaded_file and client:
            try:
                client.files.delete(name=uploaded_file.name)
                logger.debug("Remote Gemini uploaded asset deleted: %s", uploaded_file.name)
            except Exception as clean_err:
                logger.warning("Failed to delete remote uploaded file %s: %s", uploaded_file.name, clean_err)

# --------------------------------------------------------------------------------------------------
# STEP 8: STATIC ASSET & SINGLE-PAGE APPLICATION (SPA) ROUTING
# --------------------------------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
@app.get("/login", response_class=HTMLResponse)
@app.get("/signup", response_class=HTMLResponse)
@app.get("/history", response_class=HTMLResponse)
def read_index():
    """Serves the main single-page application frontend."""
    try:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html"),
            os.path.join(os.path.dirname(__file__), "frontend", "index.html"),
            os.path.join(os.getcwd(), "frontend", "index.html"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html"),
            os.path.join(os.path.dirname(__file__), "index.html"),
            os.path.join(os.getcwd(), "index.html"),
            "frontend/index.html",
            "index.html",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
        logger.warning("index.html not found in searched paths: %s", possible_paths)
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)
    except Exception as e:
        logger.error("Error loading index.html: %s", e, exc_info=True)
        return HTMLResponse(content=f"Error loading index.html: {str(e)}", status_code=500)

@app.get("/static/{file_path:path}")
def serve_static(file_path: str):
    """Serves static assets such as images and icons."""
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", file_path),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", file_path),
        os.path.join(os.getcwd(), "static", file_path),
        os.path.join(os.getcwd(), "api", "static", file_path),
        os.path.join("/var/task", "static", file_path),
        os.path.join("/var/task", "api", "static", file_path),
    ]
    for p in possible_paths:
        if os.path.exists(p) and os.path.isfile(p):
            return FileResponse(p)
    logger.warning("Static file not found: %s. Searched: %s", file_path, possible_paths)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/health")
@app.get("/health")
def health():
    """Healthcheck endpoint for deployment monitoring."""
    logger.debug("Healthcheck requested")
    return {"status": "healthy", "service": "Social Caption Generator API with Auth (Google Gemini)"}
