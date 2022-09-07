import json
import base64
import boto3

runtime = boto3.client('runtime.sagemaker')
ENDPOINT = "image-classification-2022-09-06-13-13-30-012"

def lambda_handler(event, context):
    image = base64.b64decode(event['image_data'])
    reponse = response = runtime.invoke_endpoint(
                                        EndpointName=ENDPOINT,
                                        ContentType='application/x-image',
                                        Body=image)
    inferences = response['Body'].read().decode('utf-8')
    event["inferences"] = inferences
    return {
        'statusCode': 200,
        'body': {
            "image_data": event['image_data'],
            "s3_bucket": event['s3_bucket'],
            "s3_key": event['s3_key'],
            "inferences": event['inferences'],
        }
    }