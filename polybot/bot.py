import os
import telebot
from loguru import logger
import time
from telebot.types import InputFile
import boto3
import requests
import json
class Bot:
    def __init__(self, token, telegram_chat_url):
        self.telegram_bot_client = telebot.TeleBot(token)
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)
        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')
    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)
    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)
    @staticmethod
    def is_current_msg_photo(msg):
        return 'photo' in msg
    def download_user_photo(self, msg):
        if not self.is_current_msg_photo(msg):
            raise RuntimeError('Message content of type "photo" expected')
        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        # Define the base directory where you want to store your photos
        base_dir = 'photos'
        # Combine the base directory with the file path from Telegram
        local_file_path = os.path.join(base_dir, file_info.file_path)
        # Extract the directory path
        local_dir = os.path.dirname(local_file_path)
        # Create the directory if it does not exist
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        # Save the file
        with open(local_file_path, 'wb') as photo:
            photo.write(data)
        return local_file_path
    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")
        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')
class QuoteBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        if msg["text"] != "Please don't quote me":
            self.send_text_with_quote(msg['chat']['id'], msg["text"], quoted_msg_id=msg["message_id"])
class ObjectDetectionBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        if 'text' in msg:
            self.send_text(msg['chat']['id'], f"Your original message: {msg['text']}")
        elif 'photo' in msg:
            images_bucket = os.environ['BUCKET_NAME']
            s3_client = boto3.client('s3')
            img_path = self.download_user_photo(msg)
            logger.info("---------------------------------------")
            logger.info(img_path)
            photo_key = os.path.basename(img_path)
            try:
                s3_client.upload_file(img_path, images_bucket, photo_key)
            except Exception as e:
                logger.error(f'Error uploading photo to S3: {e}')
                return
            logger.info(f'Photo uploaded successfully to S3 bucket')
            self.process_image_with_yolo5(msg['chat']['id'], photo_key)
    def count_objects(self,data):
      counts = {}
      data = json.loads(data)
      # Iterate through labels in the JSON data
      for label in data['labels']:
        # Get the class label
        obj_class = label['class']
        # Increment count for the class label
        counts[obj_class] = counts.get(obj_class, 0) + 1
      result = ''
      for obj_class, count in counts.items():
        result += f'{obj_class}: {count}\n'
      return result
    def process_image_with_yolo5(self, chat_id, photo_key):
        url = "http://yolo5-app:8081/predict"
        params = {"imgName": photo_key}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        for attempt in range(5):  # Retry up to 5 times
            try:
                response = requests.post(url, params=params, headers=headers)
                if response.status_code == 200:
                    prediction_img_path = response.content.decode("utf-8")
                    if prediction_img_path:
                        result = self.count_objects(prediction_img_path)
                        self.send_text(chat_id, result)
                    else:
                        self.send_text(chat_id, "No prediction image available")
                    break  # Exit the loop if successful
                else:
                    logger.error(f'Error from yolo5 service: {response.status_code}')
            except requests.exceptions.RequestException as e:
                logger.error(f'Attempt {attempt + 1}: Error sending request to yolo5 service: {e}')
            time.sleep(2 ** attempt)  # Exponential backoff
        else:  # If all retries fail
            self.send_text(chat_id, "Failed to process the image with yolo5 service.")
