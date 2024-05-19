
from openai import OpenAI
import json

API_KEY = 'sk-65359a75e7554786ae239db43d3e7ab9'

if __name__ == '__main__':

    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

    system_prompt = "You are an experienced content writer with a deep understanding of different types of articles and content, and you are able to fairly judge their quality. " \
                    "I will provide you with an incomplete poem, article, or other type of text, and the ellipsis at the end represents that it needs to be completed. I will also give you some completion results, and you need to evaluate and rank the quality of these completion results. For example, if I give you three completion results, and you think the third one is better than the first one, and the first one is better than the second one, then you should return '3>1>2', and you can also provide some comments." \
                    "Please reply in the format of 'Ranking: x>x>x \n Comments:'"

    with open('prompt.json', 'r') as file:
        prompt_data = json.load(file)

    for data in prompt_data:
        user_prompt = f"This is a {data['type']}." \
            f"Original content:\n" \
            f"{data['content']}\n"
        for i, completion in enumerate(data['answers']):
            user_prompt += f"Completion result {i + 1}:\n" \
                        f"{completion}\n"
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False
        )

        print(response.choices[0].message.content)