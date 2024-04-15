
import json

import json
import boto3
import base64
import urllib
import os

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    s3_input_uri=event['s3key1']
    
    key = '/'.join(s3_input_uri.split('/')[1:])
    # ## TODO: fill in
    bucket =s3_input_uri.split('/')[0]

    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    s3.download_file(bucket, key, '/tmp/image.png')
    # We read the data from a file
    with open('/tmp/image.png', "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
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
	
	#___________________________________________________________________--
	import json
# import sagemaker
import base64
import boto3

# from sagemaker.serializers import IdentitySerializer
 
# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2022-10-27-13-54-52-202"
seg=boto3.client('runtime.sagemaker')
def lambda_handler(event, context):

    image = base64.b64decode(event['body']['image_data'])
    print(base64.b64decode(event['body']['image_data']))
    res=seg.invoke_endpoint(EndpointName=ENDPOINT,ContentType='application/x-image',Body=image)
    inferences = res['Body'].read()
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event),
        'inferences':event["inferences"]
    }

	
		#___________________________________________________________________--
import json
THRESHOLD = .90
def lambda_handler(event, context):
    # Grab the inferences from the event
    inferences = event["inferences"]

    i = inferences.index(',')
    val1, val2 = inferences[1:i], inferences[i+1:-1]

    flag=0
    if((float(val1)>THRESHOLD) or (float(val2)>THRESHOLD)): 
        flag=1
    else:
        flag=0
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = flag

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
