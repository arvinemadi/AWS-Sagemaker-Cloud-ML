import json
import boto3
import base64
import botocore

s3 = boto3.resource('s3')

def lambda_handler(event, context):
    key = event['s3_key'] 
    bucket = event['s3_bucket']            
    
    try:
        s3.Bucket(bucket).download_file(key, '/tmp/image.png')
    except botocore.exceptions.ClientError as error:
        print("File not accessible")
        raise
    
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }