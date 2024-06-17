import time
from pathlib import Path
from flask import Flask, request, jsonify

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../yolov5')))


from detect import run
import uuid
import yaml
from loguru import logger
import os
import boto3
import pymongo


images_bucket = os.environ['BUCKET_NAME']
# Initialize the S3 client
s3_client = boto3.client('s3')

with open("coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']
    
    
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())
    logger.info(f'prediction: {prediction_id}. start processing')
    # Receives a URL parameter representing the image to download from S3
    img_name = request.args.get('imgName')
    # TODO download img_name from S3, store the local image path in original_img_path
    #  The bucket name should be provided as an env var BUCKET_NAME.
    # Local file path to save the downloaded image
    original_img_path = str(img_name)
    # Download the image from the S3 bucket to the original img path
    try:
        s3_client.download_file(images_bucket, img_name, original_img_path)
        logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
    except Exception as e:
        logger.error(f'Error downloading image from S3:{e}')
        return 'Error downloading image from S3'
    logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )
    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')
    # This is the path for the predicted image with labels
    # The predicted image typically includes bounding boxes drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')
    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
    s3_key = f'predictions/{img_name}'
    try:
        s3_client.upload_file(str(predicted_img_path), images_bucket, s3_key)
    except Exception as e:
        logger.error(f'Error uploading predicted image to S3:{e}')
        return 'Error uploading predicted image to S3'
    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]
        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')
        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'labels': labels,
            'time': time.time()
        }
        # TODO store the prediction_summary in MongoDB
        # First, establish connection to MongoDB
        # client = pymongo.MongoClient("mongodb://localhost:27017/")
        client = pymongo.MongoClient("mongodb://mongo1:27017,mongo2:27017,mongo3:27017/?replicaSet=rs0")
        db = client['mongoDB']
        collection = db['Yolo_Prediction']
        # Insert the prediction_summary document into MongoDB collection
        collection.insert_one(prediction_summary)
        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary stored in MongoDB')
        prediction_summary["_id"] = str(prediction_summary["_id"])
        return jsonify(prediction_summary)
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404
@app.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
