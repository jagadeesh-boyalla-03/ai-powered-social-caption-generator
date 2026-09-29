import os
import io
import re
import json
import tempfile
import shutil
import random
import time
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="AI Social Media Caption API (Powered by Google Gemini)")

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    if not key:
        return False
    return len(key.strip()) >= 20

def clamp_caption(text: str, max_chars: int) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text.strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."

def build_fallback_caption(platform_key: str, description: str, tone: str, context: str) -> str:
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
    last_exception = None
    for model_name in GEMINI_MODEL_FALLBACKS:
        for attempt in range(max_retries):
            try:
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
                    time.sleep(wait_time)
                else:
                    break
    if last_exception:
        raise last_exception
    raise Exception("Gemini service unavailable after attempting fallback models.")

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
def read_index():
    try:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html"),
            os.path.join(os.path.dirname(__file__), "index.html"),
            os.path.join(os.getcwd(), "index.html"),
            "index.html",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"Error loading index.html: {str(e)}", status_code=500)

@app.get("/api/health")
@app.get("/health")
def health():
    return {"status": "healthy", "service": "Social Caption Generator API (Google Gemini)"}

@app.post("/{full_path:path}")
@app.post("/")
async def generate_captions(
    file: UploadFile = File(...),
    platforms: str = Form(...),
    tone: str = Form("Friendly and upbeat"),
    context: str = Form(""),
    api_key: str = Form("")
):
    temp_file_path = None
    uploaded_file = None
    client = None
    try:
        # Determine API Key: use user-supplied or fallback to environment variable
        key = api_key.strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not key:
            raise HTTPException(
                status_code=400,
                detail="Gemini API key is missing. Add GEMINI_API_KEY to your .env file or server environment."
            )
        if not looks_like_gemini_api_key(key):
            raise HTTPException(
                status_code=400,
                detail="Invalid API key format. Please check your Gemini API key."
            )

        client = genai.Client(api_key=key)

        is_video = file.content_type and file.content_type.startswith("video/")
        media_type = "video" if is_video else "image"

        if is_video:
            file_extension = os.path.splitext(file.filename)[1] if file.filename else ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name

            uploaded_file = client.files.upload(file=temp_file_path)
            max_polls = 45
            poll_count = 0
            while uploaded_file.state.name == "PROCESSING" and poll_count < max_polls:
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
                poll_count += 1

            if uploaded_file.state.name == "FAILED":
                raise Exception("Video processing failed on Gemini.")
            
            desc_resp = generate_gemini_content_with_retry(
                client=client,
                contents=[uploaded_file, "Provide a plain, factual, one-sentence description of this video."]
            )
            image_description = desc_resp.text.strip()
        else:
            image_bytes = await file.read()
            raw_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            desc_resp = generate_gemini_content_with_retry(
                client=client,
                contents=[raw_img, "Provide a plain, factual, one-sentence description of this image."]
            )
            image_description = desc_resp.text.strip()

        # Generate Platform Captions
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

        caption_text_map = {}
        fallback_used = False
        try:
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
            print(f"Error generating structured captions: {caption_error}")
            fallback_used = True
            caption_text_map = {}

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

        return {
            "description": image_description,
            "captions": results,
            "fallback_used": fallback_used
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()

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
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        if uploaded_file and client:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
