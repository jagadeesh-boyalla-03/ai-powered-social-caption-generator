"""
====================================================================================================
PROJECT: AI-Powered Multi-Platform Social Media Caption Generator (CLI Edition)
POWERED BY: Google Gemini Multimodal Vision-Language AI
====================================================================================================
This script demonstrates the end-to-end command-line process:
  - STEP 1: Load environment variables and parse CLI arguments (image path, platforms, tone, context).
  - STEP 2: Configure platform-specific character limits and hashtag requirements.
  - STEP 3: Perform Multimodal Vision analysis on the image to generate a factual scene description.
  - STEP 4: Engineer prompt guidelines tailored to each requested social media platform.
  - STEP 5: Call Gemini LLM to generate platform-ready captions and display formatted results.
====================================================================================================
Usage:
    python 6_social_media_captions.py test.jpg
    python 6_social_media_captions.py test.jpg --tone playful --platforms instagram,x,linkedin
====================================================================================================
"""

import sys
import os
import logging
import argparse
from PIL import Image
from google import genai
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Ensure proper Unicode UTF-8 character encoding in Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Configure logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format=LOG_FORMAT
)
logger = logging.getLogger("social_media_captions_cli")

# --------------------------------------------------------------------------------------------------
# STEP 1: LOAD ENVIRONMENT CONFIGURATION
# --------------------------------------------------------------------------------------------------
load_dotenv()

# --------------------------------------------------------------------------------------------------
# STEP 2: PLATFORM COPYWRITING CONSTRAINTS & RULES
# --------------------------------------------------------------------------------------------------
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
}

# --------------------------------------------------------------------------------------------------
# STEP 3: MULTIMODAL VISION SCENE EXTRACTION
# --------------------------------------------------------------------------------------------------
def get_image_description(client: genai.Client, image_path: str) -> str:
    """
    Opens the input image using Pillow and passes it to Google Gemini 3.5 Flash Lite
    to extract a factual, high-level summary of what is happening in the picture.
    """
    logger.info("Opening image at path: %s", image_path)
    raw_image = Image.open(image_path).convert("RGB")
    logger.info("Sending image to Gemini Vision for scene description...")
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[raw_image, "Provide a plain, factual, one-sentence description of this image."]
    )
    desc = response.text.strip()
    logger.info("Scene description received: '%s'", desc)
    return desc

# --------------------------------------------------------------------------------------------------
# STEP 4: PROMPT ENGINEERING FOR TARGET PLATFORMS
# --------------------------------------------------------------------------------------------------
def build_prompt(image_description: str, platform_key: str, tone: str, extra_context: str) -> str:
    """
    Constructs an optimized copywriting instruction for Gemini including:
    - Extracted media scene description
    - User-provided context / campaign notes
    - Specific character limits and hashtag requirements
    """
    rules = PLATFORM_RULES[platform_key]
    context_line = f"\nAdditional context from user: {extra_context}\n" if extra_context else ""

    prompt = f"""You are an elite social media copywriter.
Media description: "{image_description}"
{context_line}
Platform: {rules['label']}
Platform style requirements: {rules['style']}
Character Limit: strictly under {rules['max_chars']} characters

Requirements:
- Tone: {tone}
- Hashtags: Include 5+ relevant, high-traffic hashtags on the final line(s).
- Structure: Punchy hook, concise body/takeaway, quick call to action, followed by hashtags.
- Output ONLY the ready-to-post caption text.
"""
    return prompt

# --------------------------------------------------------------------------------------------------
# STEP 5: AI CAPTION GENERATION
# --------------------------------------------------------------------------------------------------
def generate_caption_for_platform(client: genai.Client, image_description: str, platform_key: str, tone: str, extra_context: str) -> str:
    """Sends the engineered prompt to Gemini and returns the final caption text."""
    prompt = build_prompt(image_description, platform_key, tone, extra_context)
    logger.debug("Generating caption for platform=%s, tone='%s'", platform_key, tone)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    return response.text.strip()

# --------------------------------------------------------------------------------------------------
# STEP 6: MAIN CLI ENTRYPOINT
# --------------------------------------------------------------------------------------------------
def main():
    # Parse CLI flags and arguments
    parser = argparse.ArgumentParser(description="Generate social media captions from an image using Google Gemini.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--platforms", default="instagram,x,facebook,linkedin,pinterest",
                         help="Comma-separated list of platforms (default: all)")
    parser.add_argument("--tone", default="friendly and upbeat",
                         help="Overall tone, e.g. 'playful', 'professional', 'inspirational', 'humorous'")
    parser.add_argument("--context", default="",
                         help="Optional extra context, e.g. 'this is for our new product launch'")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate target platforms
    platforms = [p.strip().lower() for p in args.platforms.split(",")]
    for p in platforms:
        if p not in PLATFORM_RULES:
            logger.error("Unknown platform '%s'. Valid options: %s", p, ', '.join(PLATFORM_RULES.keys()))
            sys.exit(1)

    # Validate Gemini API Key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not found in .env.")
        sys.exit(1)

    # Initialize Google GenAI client
    client = genai.Client(api_key=api_key)

    # Execute Multimodal Vision Step
    print("\n[Step 1/2] Analyzing image with Google Gemini Vision...")
    image_description = get_image_description(client, args.image_path)
    print(f"Image context: \"{image_description}\"\n")

    # Execute Caption Generation for each platform
    print("[Step 2/2] Generating platform-specific social captions...")
    print("=" * 60)
    for platform_key in platforms:
        label = PLATFORM_RULES[platform_key]["label"]
        caption = generate_caption_for_platform(client, image_description, platform_key, args.tone, args.context)
        print(f"\n📱 {label} Caption:\n{caption}\n")
        print(f"(Length: {len(caption)} characters | Limit: {PLATFORM_RULES[platform_key]['max_chars']})")
        print("-" * 60)

if __name__ == "__main__":
    main()
