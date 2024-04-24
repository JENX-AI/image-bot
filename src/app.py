# -*- coding: utf-8 -*-
import logging
import os
from dotenv import load_dotenv
import slack_bolt
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from utils.image_model import generate_image
from utils.logger import get_logger
from utils.regex_funcs import regex_handler
# Credentials
load_dotenv()
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
TOGETHER_API_KEY = os.environ["TOGETHER_API_KEY"]

app = App(token=SLACK_BOT_TOKEN)
client = WebClient(SLACK_BOT_TOKEN)
logger = get_logger(__name__)

# ====================================
# Slack event listeners
# ====================================

@app.message(".*")
def message_handler(message: dict, say: slack_bolt.Say, logger: logging.Logger) -> None:
    """
    Handles messages to chatbot within Apps
    
    Parameters
    ----------
    message : dict
        message received from Slack user in Apps
    
    say : slack_bolt.Say
        sends message to Slack
    
    logger : logging.Logger
        logging object
    """
    try:
        # Extract message and metadata
        user_message = message["text"]
        destination = message["channel"]
        message_id = message["client_msg_id"]
        # Format prompt
        prompt = regex_handler(user_message)
        # Get and save image
        filename = generate_image(prompt, message_id)
        
        app.client.files_upload_v2(
            channel=destination,
            file=filename,
        )
        # Delete image file after uploading
        os.remove(filename)
        
    except Exception as e:
        logger.error(f"Error: {e}")


@app.event(("app_mention"))
def handle_app_mention_events(body, say, logger):
    """
    Handles direct (@) messages to chatbot within Channels
    
    Parameters
    ----------
    body : dict
        message received from Slack user in Channels
    
    say : slack_bolt.Say
        sends message to Slack
    
    logger : logging.Logger
        logging object
    """
    try:
        # Extract message and metadata
        user_message = body["event"]["text"]
        message_id = body["event"]["client_msg_id"]
        destination = body["event"]["channel"]
        # Format prompt
        prompt = regex_handler(user_message)
        # Get and save image
        filename = generate_image(prompt, message_id)
        
        app.client.files_upload_v2(
            channel=destination,
            file=filename,
        )
        # Delete image file after uploading
        os.remove(filename)
    
    except Exception as e:
        logger.error(f"Error: {e}")
    

# ====================================
# Initialisation
# ====================================

if __name__ == "__main__":
    # Create an app-level token with connections:write scope
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
