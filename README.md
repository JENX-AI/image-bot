# image-bot

Image-generating LLM bots for integration with Slack.

![](https://img.shields.io/badge/Amazon%20AWS-232F3E.svg?style=flat&logo=Amazon-AWS&logoColor=white)
![](https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white)
![](https://img.shields.io/badge/Slack-4A154B.svg?style=flat&logo=Slack&logoColor=white)

## Models

Source: [together.ai](https://docs.together.ai/docs/inference-models)

| Organization | Model Name            | Model String for API               | Context length | Type  |
| ------------ | --------------------- | ---------------------------------- | -------------- | ----- |
| 01.AI        | 01-ai Yi Chat (34B)   | zero-one-ai/Yi-34B-Chat            | 4096           | Chat  |
| Meta         | LLaMA-2 Chat (13B)    | meta-llama/Llama-2-13b-chat-hf     | 4096           | Chat  |
| mistralai    | Mistral (7B) Instruct | mistralai/Mistral-7B-Instruct-v0.1 | 8192           | Chat  |
| Stanford     | Alpaca (7B)           | togethercomputer/alpaca-7b         | 2048           | Chat  |
| Stability AI | Stable Diffusion 2.1  | stabilityai/stable-diffusion-2-1   | N/A            | Image |

## Directory structure

The current structure for hosting four baseline chat models is as follows:

| Parent directory | Model directory                | Repo directory    |
| ---------------- | ------------------------------ | ----------------- |
| `slackbot-repos` |                                |                   |
|                  | `alpaca-7b`                    | `slackbot`        |
|                  | `Llama-2-13b-chat-hf`          | `slackbot`        |
|                  | `Mistral-7B-Instruct-v0.1`     | `slackbot`        |
|                  | `Yi-34B-Chat`                  | `slackbot`        |
|                  | `Intro-bot`                    | `slack_intro_bot` |
|                  | `stable-diffusion-xl-base-1.0` | `image-bot`       |

## Slack OAuth Scopes

The following bot scopes are required for this implementation:

| OAuth Scope       | Scope Type | Description                                                                   |
| ----------------- | ---------- | ----------------------------------------------------------------------------- |
| app_mentions:read | Bot        | View messages that directly mention @app in conversations that the app is in  |
| chat:write        | Bot        | Send messages as @app                                                         |
| chat:write:public | Bot        | Send messages to channels @app isn't a member of                              |
| files:write       | Bot        | Upload, edit, and delete files as app                                         |
| im:history        | Bot        | View messages and other content in direct messages that app has been added to |
| connections:write | App        | Route your app’s interactions and event payloads over WebSockets              |
