import os
import base64
import time
from template import prompt_it, prompt_text
import json
import csv
import openai

from dotenv import load_dotenv
load_dotenv()

client = openai.OpenAI()
    

class TriplePromptManager:
    def __init__(self, triple_count=5, template_str=prompt_it):
        self.triple_count = triple_count
        self.template_str = template_str

    def build_prompt(self, title_list):
        history_str = ""
        for idx, title in enumerate(title_list, 1):
            history_str += f"{idx}: \"{title}\"\n"
        return self.template_str.format(history=history_str.strip(), triple_count=self.triple_count)


class TripleExtractor:
    def __init__(self, model, temperature=0.3, api_key=None):
        self.model = model
        self.temperature = temperature
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            raise ValueError("OpenAI API key must be set via argument or environment variable")


    def encode_image(self, image_path):
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


    def extract(self, news_id_list, prompt, max_retries=3):
        image_inputs = []

        for news_id in news_id_list:
            encoded_image = self.encode_image(f"newsImages/{news_id}.jpg")
            image_inputs.append({"type": "image_url", "image_url": {"url": encoded_image}})

        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts structured triples from image-title pairs and should convert it into the given structure."},
            {"role": "user", "content": [
                *image_inputs,
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # pprint.pprint(messages)

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[Retry {attempt+1}/{max_retries}] Error: {e}")
                time.sleep(2)

        raise RuntimeError("Failed to get response from OpenAI after several attempts.")



def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_history_list(behavior_path):
    history_list = []
    with open(behavior_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line.strip())
            if "history" in row:
                history_list.append(row["history"])
    return history_list


def load_news_info(news_info_path):
    news_info = {}
    
    with open(news_info_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) > 4:
                id = row[0]
                title = row[3]
                news_info[id] = title
    
    return news_info

def extract_batch_triples(
    model,
    jsonl_path,
    triple_count=5,
    temperature=0.3,
    api_key=None,
    save_output=False,
):
    prompt_manager = TriplePromptManager(triple_count=triple_count)
    extractor = TripleExtractor(model=model, temperature=temperature, api_key=api_key)
    
    behavior_list = load_jsonl(jsonl_path)
    # print(behavior_list[0]['history'].split())

    nid2title = load_news_info("MyMINDsmall/test/news.tsv")
    
    results = []

    for idx, behavior in enumerate(behavior_list, 1000):
        if idx > 1004:
            break

        news_id_list = behavior['history'].split()
        title_list = []

        for news_id in news_id_list:
            title = nid2title[news_id]

            if not title:
                print(f"[Warning] Skipping news_id {news_id} due to missing title.")
                continue

            title_list.append(title)

        try:
            prompt = prompt_manager.build_prompt(title_list)
            triples = extractor.extract(news_id_list, prompt)
            time.sleep(3)

            result = {
                "triples": triples
            }
        except Exception as e:
            print(f"[Error] Failed at index {idx}: {e}")
            result = {
                "triples": [],
                "error": str(e)
            }

        results.append(result)

    # ✅ .json 파일 저장
    if save_output:
        output_path = "./multimodal_triples.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)  # indent for pretty print
        print(f"\n[✔] Results saved to {output_path}")

    return results
