"""
Generate ready-to-post social media captions from an image, for
multiple platforms at once (Instagram, X/Twitter, Facebook, LinkedIn, Pinterest) using Google Gemini.

Before running, ensure your Gemini API key is set in .env or as an environment variable:
    Windows (PowerShell):  $env:GEMINI_API_KEY="your-key"
    Mac/Linux:             export GEMINI_API_KEY="your-key"

Usage:
    python 6_social_media_captions.py path/to/image.jpg
    python 6_social_media_captions.py path/to/image.jpg --tone playful --platforms instagram,x,linkedin
"""

import sys
import os
import argparse
from PIL import Image
from google import genai
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

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

def get_image_description(client: genai.Client, image_path: str) -> str:
    raw_image = Image.open(image_path).convert("RGB")
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=[raw_image, "Provide a plain, factual, one-sentence description of this image."]
    )
    return response.text.strip()

def build_prompt(image_description: str, platform_key: str, tone: str, extra_context: str) -> str:
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

def generate_caption_for_platform(client: genai.Client, image_description: str, platform_key: str, tone: str, extra_context: str) -> str:
    prompt = build_prompt(image_description, platform_key, tone, extra_context)
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    return response.text.strip()

def main():
    parser = argparse.ArgumentParser(description="Generate social media captions from an image using Google Gemini.")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--platforms", default="instagram,x,facebook,linkedin,pinterest",
                         help="Comma-separated list of platforms (default: all)")
    parser.add_argument("--tone", default="friendly and upbeat",
                         help="Overall tone, e.g. 'playful', 'professional', 'inspirational', 'humorous'")
    parser.add_argument("--context", default="",
                         help="Optional extra context, e.g. 'this is for our new product launch'")
    args = parser.parse_args()

    platforms = [p.strip().lower() for p in args.platforms.split(",")]
    for p in platforms:
        if p not in PLATFORM_RULES:
            print(f"Unknown platform '{p}'. Valid options: {', '.join(PLATFORM_RULES.keys())}")
            sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not found in .env.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    image_description = get_image_description(client, args.image_path)
    print(f"\nImage description (from Gemini): {image_description}\n")

    print("=" * 60)
    for platform_key in platforms:
        label = PLATFORM_RULES[platform_key]["label"]
        caption = generate_caption_for_platform(client, image_description, platform_key, args.tone, args.context)
        print(f"\n📱 {label} caption:\n{caption}\n")
        print(f"(length: {len(caption)} characters, limit: {PLATFORM_RULES[platform_key]['max_chars']})")
        print("-" * 60)

if __name__ == "__main__":
    main()
