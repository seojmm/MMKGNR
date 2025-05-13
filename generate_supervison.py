import os
import base64
import time
from templates import prompt_text
import json
import csv
import openai
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

load_dotenv()

client = openai.OpenAI()


class TriplePromptManager:
    def __init__(self, triple_count=10, template_str=prompt_text):
        self.triple_count = triple_count
        self.template_str = template_str

    def build_prompt(self, title):
        return self.template_str.format(triple_count=self.triple_count, title=title)


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
    

    def extract_triples(self, triple_count, news_id, title, max_retries=3):
        prompt_manager = TriplePromptManager(triple_count)
        prompt = prompt_manager.build_prompt(title)
        
        messages = [
            {"role": "system", "content": "You are a triple extractor for constructing user behavior knowledge graphs with factual triples from news titles and image."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt.format(triple_count=triple_count, title=title)},
                {"type": "image_url", "image_url": {"url": self.encode_image(f"datasets/newsImages/{news_id}.jpg")}},
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


# def get_id2title(news_info_path):
#     news_info = {}
    
#     with open(news_info_path, 'r', encoding='utf-8') as f:
#         reader = csv.reader(f, delimiter='\t')
#         for row in reader:
#             if len(row) > 4:
#                 id = row[0]
#                 title = row[3]
#                 news_info[id] = title
    
#     return news_info


def load_news(file_path, sample_size=None):
    df = pd.read_csv(file_path, sep='\t', header=None, usecols=[0, 3], names=['news_id', 'title'])

    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    return list(df.itertuples(index=False, name=None))



def extract_supervision(
    news_data_path,
    model="gpt-4o-mini",
    triple_count=10,
    temperature=0.3,
    api_key=None,
    save_output=False,
):
    extractor = TripleExtractor(model=model, temperature=temperature, api_key=api_key)
    
    news_data_list = load_news(news_data_path)
    
    results = []

    for news_id, title in tqdm(news_data_list, total=len(news_data_list), desc="Extracting image-guided triples"):
        try:
            triples = extractor.extract_triples(news_id=news_id, title=title, triple_count=10)
            results.append({
                "news_id": news_id,
                "title": title,
                "triples": triples.split("\n")
            })
        except Exception as e:
            print(f"Failed to process {news_id}: {e}")


    if save_output:
        current_time = datetime.now().strftime("%y%m%d_%H%M%S")
        os.makedirs("supervision", exist_ok=True)  # ✅ supervision 디렉토리 생성
        output_path = f"supervision/{current_time}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\nResults saved to {output_path}")

    return results

if __name__ == "__main__":
    # print(load_news("datasets/MyMINDsmall/train/news.tsv"))
    results = extract_supervision(
        news_data_path="datasets/MyMINDsmall/train/news.tsv",
        triple_count=10,
        save_output=True,
    )

    print(results[:10])