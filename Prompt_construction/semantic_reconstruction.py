from openai import OpenAI
import jsonlines
from tqdm import tqdm
import os

os.environ["OPENAI_API_KEY"] = "" 
client = OpenAI()
model_name = "gpt-4o"  

prompt_template = """
You are a prompt refiner.
Combine the following five elements into one fluent, natural, and descriptive sentence
that can be used as a text-to-video generation prompt.

Scene: "{scene}"
Subject: "{subject}"
Object: "{object}"
Action: "{action}"
Mood: "{mood}"

Output only the final merged sentence.
"""

input_path = r""
output_path = r""

with jsonlines.open(input_path) as reader:
    data = [obj for obj in reader]

with jsonlines.open(output_path, mode="w") as writer:
    for item in tqdm(data):
        scene = item.get("scene")
        subject = item.get("subject")
        obj = item.get("object")
        action = item.get("action")
        mood = item.get("mood")

        prompt = prompt_template.format(scene=scene, subject=subject, object=obj, action=action, mood=mood)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a precise and descriptive writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_completion_tokens=256
            )

            result = response.choices[0].message.content.strip()
            writer.write({"scene": scene, "merged_prompt": result})
            print(result)

        except Exception as e:
            print(f"Error for {scene}: {e}")
            writer.write({"scene": scene, "merged_prompt": None})

print(f"Done! Saved to {output_path}")
