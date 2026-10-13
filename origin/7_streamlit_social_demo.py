"""
Client-facing demo UI — upload an image, pick platforms and tone,
get ready-to-post captions for all of them at once using Google Gemini.

Run:
    streamlit run 7_streamlit_social_demo.py
"""

import os
import time
import random
import streamlit as st
from PIL import Image
from google import genai
from dotenv import load_dotenv

# Load API key from .env file automatically
load_dotenv()

PLATFORM_RULES = {
    "Instagram": {
        "key": "instagram",
        "max_chars": 2200,
        "style": "warm, visual, engaging; 5-8 relevant hashtags; 1-2 emojis; call to action",
    },
    "X (Twitter)": {
        "key": "x",
        "max_chars": 280,
        "style": "punchy, witty, very concise; 2-3 hashtags; under 280 characters total",
    },
    "Facebook": {
        "key": "facebook",
        "max_chars": 600,
        "style": "conversational, community-focused, storytelling; 4-5 relevant hashtags",
    },
    "LinkedIn": {
        "key": "linkedin",
        "max_chars": 750,
        "style": "professional, insight-driven, practical career/business takeaways; 5+ hashtags",
    },
    "Pinterest": {
        "key": "pinterest",
        "max_chars": 500,
        "style": "descriptive, keyword-rich, inspirational; 5-7 hashtags",
    },
}

DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.6-flash"]

st.set_page_config(page_title="Social Media Caption Generator", layout="centered")
st.title("📱 Social Media Caption Generator")
st.write("Upload an image → get ready-to-post captions for every platform.")

# API Key Validation and Client Initialization
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

if not api_key:
    st.sidebar.warning("⚠️ GEMINI_API_KEY environment variable not found in .env.")
    st.info("👉 Please add `GEMINI_API_KEY` to your `.env` file or enter it in the sidebar to generate captions.")
    st.stop()

client = genai.Client(api_key=api_key.strip())

def generate_content_with_retry(client, contents, max_retries=3):
    last_err = None
    for model_name in DEFAULT_MODELS:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents
                )
                return response.text.strip()
            except Exception as e:
                last_err = e
                err_str = str(e).upper()
                is_transient = "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str
                if is_transient and attempt < max_retries - 1:
                    time.sleep((1.5 ** attempt) + random.uniform(0.5, 1.0))
                else:
                    break
    if last_err:
        raise last_err
    raise Exception("Gemini service temporarily unavailable.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)
with col1:
    selected_platforms = st.multiselect(
        "Platforms", list(PLATFORM_RULES.keys()), default=["LinkedIn", "Instagram", "X (Twitter)"]
    )
with col2:
    tone = st.selectbox(
        "Tone", ["Friendly and upbeat", "Playful", "Professional", "Inspirational", "Humorous", "Minimal/aesthetic"]
    )

extra_context = st.text_input("Optional context (e.g. 'new product launch', 'weekend getaway')", "")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Captions", type="primary", disabled=len(selected_platforms) == 0):
        try:
            with st.spinner("Analyzing image using Gemini Vision..."):
                image_description = generate_content_with_retry(
                    client=client,
                    contents=[image, "Provide a plain, factual, one-sentence description of this image."]
                )

            st.caption(f"Image understood as: *{image_description}*")

            for platform_name in selected_platforms:
                rules = PLATFORM_RULES[platform_name]
                context_line = f"\nAdditional context: {extra_context}\n" if extra_context else ""

                prompt = f"""You are an elite social media copywriter.
Media Description: "{image_description}"
{context_line}
Platform: {platform_name}
Platform Requirements: {rules['style']}
Character Limit: strictly under {rules['max_chars']} characters

Requirements:
- Tone: {tone}
- Hashtags: Include 5+ relevant, high-traffic hashtags on the final line(s).
- Structure: Punchy hook, concise body/takeaway, quick call to action, followed by hashtags.
- Output ONLY the ready-to-post caption text without preambles or markdown code fences.
"""
                with st.spinner(f"Writing {platform_name} caption..."):
                    caption = generate_content_with_retry(
                        client=client,
                        contents=prompt
                    )

                st.subheader(platform_name)
                st.text_area(f"{platform_name} caption", caption, height=140, label_visibility="collapsed")
                st.caption(f"Length: {len(caption)} / {rules['max_chars']} characters")
                st.divider()
        except Exception as e:
            st.error(f"Error generating captions: {str(e)}")
