import os
from typing import Dict, List

from openai import OpenAI

I8K_AI_TOOLS_PRODUCT_ID = os.environ.get("I8K_AI_TOOLS_PRODUCT_ID")
I8K_AI_TOOLS_LLM_API_KEY = os.environ.get("I8K_AI_TOOLS_LLM_API_KEY")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "")

client = OpenAI(
        base_url=f"https://api.infomaniak.com/1/ai/{I8K_AI_TOOLS_PRODUCT_ID}/openai",
        # This is the default and can be omitted
        api_key=I8K_AI_TOOLS_LLM_API_KEY,
        )

completion = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {
                "role":    "system",
                "content": get_prompt_translation("system_prompt", lang)
                },
            {"role": "user", "content": get_prompt_translation("body", lang, body=body, nb_groups=len(groups))}
            ]
        )