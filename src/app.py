import os
# import together
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack import WebClient
from langchain_together import Together as togetherLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

import base64
import io
from PIL import Image

from langchain import LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from together import Together

#Credentials
load_dotenv()
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]
# Event API & Web API
app = App(token=SLACK_BOT_TOKEN)
client = WebClient(SLACK_BOT_TOKEN)


template = """You are a general purpose chatbot. Try explaining in simple words. Answer in less than 100 words. If you don't know the answer simply respond as Based on the information it's unclear.

{history}

Human: {human_input}
Assistant: """

prompt = PromptTemplate(
    input_variables=["history", "human_input"],
    template=template
)

llm = togetherLLM(
    model="togethercomputer/Llama-2-70b-chat",
    temperature=0.7,
    max_tokens=250,
    top_k=1,
    together_api_key=TOGETHER_API_KEY
)

# together.api_key = TOGETHER_API_KEY

# Langchain implementation

chatgpt_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

def generate_image(image_description: str):
    import requests
    # print(image_description)
    endpoint = 'https://api.together.xyz/v1/completions'
    res = requests.post(endpoint, json={
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "prompt": image_description.strip(),
        "negative_prompt": "Nudity, Violence, Gore, Blood, Sex, NSFW",
        "width": 1024,
        "height": 1024,
        "steps": 40,
        "n": 4,
        "seed": 4868
    }, headers={
        "Authorization": "Bearer TOGETHER_API_KEY",
    })
    print(res)
    res = res.json()
    images = res['choices']
    for idx, b64_img in enumerate(images):
        with open(f"{image_description}_{idx}.png", "wb") as f:
            f.write(base64.b64decode(b64_img["image_base64"]))

@app.event("app_mention")
def message_handler(body, say, logger):
    user_message = body["event"]["text"]
    print(user_message)
    output_channel = body['event']['channel']
    user_message = " ".join(user_message.split(" ")[1:])
    if user_message.startswith("generate-image:"):
        prompt = "_".join(user_message.split(":")[1:]).strip()
        print(prompt)
        generate_image(prompt)
        try:
            # uploaded_file = client.files_upload_v2(file=f"{prompt}_{0}.png")
            # print(len(uploaded_file['files']))
            result = app.client.files_upload(
            channels=output_channel,
            initial_comment=f"Generated image: {prompt}_{0}",
            file=f"{prompt}_{0}.png",
        )
            result = app.client.files_upload(
            channels=output_channel,
            initial_comment=f"Generated image: {prompt}_{1}",
            file=f"{prompt}_{1}.png",
        )
            result = app.client.files_upload(
            channels=output_channel,
            initial_comment=f"Generated image: {prompt}_{2}",
            file=f"{prompt}_{2}.png",
        )
            result = app.client.files_upload(
            channels=output_channel,
            initial_comment=f"Generated image: {prompt}_{3}",
            file=f"{prompt}_{3}.png",
        )
        except Exception as e:
            print(e)
    else:
        response = chatgpt_chain.predict(human_input=user_message)
        say(response)

if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
