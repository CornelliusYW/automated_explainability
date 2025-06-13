from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import pathlib
from typing import Dict

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def generate_llm_summary(
    global_importance: Dict[str, float],
    local_explanations: Dict[int, Dict[str, float]],
    model_version: str
) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Prepare prompt parts
    parts = []

    # Add SHAP image as inline data if it exists
    shap_path = pathlib.Path("shap_summary_plot.png")
    if shap_path.exists():
        with open(shap_path, "rb") as f:
            image_data = f.read()
        parts.append(types.Part.from_bytes(
            data=image_data,
            mime_type="image/png"
        ))

    # Prepare top features text
    top_features = ", ".join(
        f"{k} ({v:.2f})"
        for k, v in sorted(global_importance.items(), key=lambda item: -abs(item[1]))[:5]
    )

    # Pick an example local explanation (first non-zero)
    example_local = next(
        (v for v in local_explanations.values() if any(vv != 0 for vv in v.values())),
        None
    )
    local_features = (
        ", ".join(f"{k}: {v:.3f}" for k, v in example_local.items())
        if example_local else "No significant local explanations available."
    )

    # Compose prompt text
    prompt_text = (
        "Provide a detailed, professional, and easily understandable summary to create explainability report based on this information: \n"
        f"Model version: {model_version}.\n"
        f"Top global feature importances: {top_features}.\n"
        f"Example local explanation (1 instance): {local_features}.\n"
        f"Explanation methods used: SHAP and/or LIME.\n"
        "Also consider the attached SHAP summary plot (if any)."
    )
    parts.append({"text": prompt_text})

    # Send to Gemini
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction="You are a master data scientist with a focus on explainability."
        )
    )
    return response.text
