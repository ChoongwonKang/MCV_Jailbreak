from openai import OpenAI
import json, re, os
from tqdm import tqdm
import jsonlines

os.environ["OPENAI_API_KEY"] = ""
client = OpenAI()
model_name = "gpt-4o"

prompt_template = """
You are a semantic decomposer.
Given a short scene or topic phrase, infer and describe four coherent elements as short phrases (2–5 words):

1. Subject — the main actor or entity involved
2. Object — the main target, entity, or focus of the action
3. Action — the key verb or interaction implied in the scene
4. Mood — the emotional tone, atmosphere, or overall feeling conveyed by the scene


Scene: "{scene}"

Output in plain text (no JSON):
Subject: ...
Object: ...
Action: ...
Mood: ...
"""

input_path = r""
output_path = r""

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with jsonlines.open(output_path, mode="w") as writer:
    for item in tqdm(data):
        scene = item["Key Phrase"]
        prompt = prompt_template.format(scene=scene)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=512,
            )

            text = response.choices[0].message.content.strip()

            def extract(key):
                m = re.search(rf"{key}\s*[:\-]\s*(.*)", text, re.IGNORECASE)
                return m.group(1).strip() if m else None

            parsed = {
                "scene": scene,
                "subject": extract("subject"),
                "object": extract("object"),
                "action": extract("action"),
                "mood": extract("mood"),
            }

            writer.write(parsed)
            print(parsed)

        except Exception as e:
            writer.write({
                "scene": scene,
                "subject": None,
                "object": None,
                "action": None,
                "mood": None
            })
            print(f"Error for {scene}: {e}")

print(f"Done! Saved to {output_path}")