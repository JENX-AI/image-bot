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
  endpoint = 'https://api.together.xyz/v1/completions'
  res = requests.post(endpoint, json={
      "model": "stabilityai/stable-diffusion-xl-base-1.0",
      "prompt": 'prompt',
      "negative_prompt": "",
      "width": 1024,
      "height": 1024,
      "steps": 40,
      "n": 4,
      "seed": 4868
  }, headers={
      "Authorization": "Bearer e259d65582550f73533932fd6ed1ed1b9046cb343ad90b097a7376bfb5ed2fdb",
  })
  res = res.json()
  print(res)
  # with open(f"{image_description}.png", "wb") as f:
  #     f.write(base64.b64decode(response.data[0].b64_json))

@app.event("app_mention")
def message_handler(body, say, logger):
    user_message = body["event"]["text"]
    print("USER MESSAGE:", user_message)
    # Cleaning the user_message
    user_message = " ".join(user_message.split(" ")[1:])
    if user_message.startswith("generate-image:"):
      prompt = "_".join(user_message.split(":")[1:])
      # response = together.Image.create(
      # prompt=prompt,
      # model="stabilityai/stable-diffusion-xl-base-1.0",
      # steps=300,
      # )
      # print(len(response["output"]['choices']))
      # # Replace with your byte stream representing the image
      # image_bytes = base64.b64decode(response['output']['choices'][0]['image_base64'])  # Binary data of your image

      # # Create a BytesIO object from the byte stream
      # # image_data = io.BytesIO(image_bytes)
      # image = Image.open(io.BytesIO(image_bytes))
      # image.save('output_image.png')
      generate_image(prompt)
      # Send the image
      # say(blocks=[
      #     {
      #         "type": "image",
      #         "image_bytes": image_bytes,
      #         "alt_text": "Example image"
      #     }
      # ])
      try:
        client.files_upload_v2(file=f"{prompt}.png")
      except Exception as e:
        print(e)
    else:
      response = chatgpt_chain.predict(human_input=user_message)
      say(response)

if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()