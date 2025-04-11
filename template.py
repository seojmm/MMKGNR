prompt_it = """# Task Description
You are given a sequence of news images first, followed by their corresponding titles.
Each image corresponds to the title at the same index.

# Input
## User's Clicked News Titles
{history}

# Reasoning Process
1. Observe each image and its corresponding title.
2. Extract exactly {triple_count} triples for each image-title pair, focus on the unique role of news image below.
    - News images emphasize entities that appear in both the news image and the title.
    - News images mitigate exaggerated expressions or sensationalized titles by depicting abstract aspects related to the topic or content.

# Output Format
Return your result as a JSON list. Each entry must be in the following structure:

[
  {{
    "title": "<title_1>",
    "triples": [
      ["head1", "relation1", "tail1"],
      ...
    ]
  }},
  ...
]

##Caution : Do not include any explanation, extra comments, or non-JSON content.
"""

prompt_text = """
You are given a sequence of news images first, followed by their corresponding titles.

Each image corresponds to the title at the same index (Image 1 → Title 1, etc.).

Your task is to extract {triple_count} factual triples from each image-title pair.

Do not rely on the title alone. You must consider the content and context of the image as the primary source of information.

Use the image to clarify, validate, or correct what the title says. If the title is exaggerated or misleading, trust the image.

Output exactly {triple_count} triples per pair in this format:
(head entity, relation, tail entity)

Separate the triples for each title with a line break (\\n).  
Do not include any explanation or extra text. Only return the triples.
"""

prompt_cot = """You will be shown a sequence of news images, followed by a list of corresponding news titles.
Each image corresponds to the title at the same index (Image 1 → Title 1, etc.).

Your task is to extract exactly {triple_count} factual knowledge triples for each image-title pair.

Proceed step by step for each pair:
1. Observe the image carefully.
2. Read the corresponding title.
3. Identify key entities, relationships, and actions from both image and title.
4. Use the image to verify or correct any exaggerated or misleading parts of the title.
5. Extract {triple_count} (subject, predicate, object) triples that reflect the core factual content.

Output Format:
- Return triples in this format: (head entity, relation, tail entity)
- Each triple must be on its own line.
- Separate the triples for each title with ".
- Do not include any explanation, commentary, or extra formatting.
"""