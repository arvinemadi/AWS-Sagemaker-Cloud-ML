import json
THRESHOLD = .93
def lambda_handler(event, context):
    
    inferences = event['inferences'][1:-1].split(',')
    inferences = [float(inference) for inference in inferences]
    meets_threshold = any (inference >= THRESHOLD for inference in inferences)
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': {
            "image_data": event['image_data'],
            "s3_bucket": event['s3_bucket'],
            "s3_key": event['s3_key'],
            "inferences": event['inferences'],
        }
    }