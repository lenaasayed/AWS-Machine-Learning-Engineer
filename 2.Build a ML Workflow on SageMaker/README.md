
# Build a ML Workflow For Scones Unlimited On Amazon SageMaker

Building an image classification model that can automatically detect which kind of vehicle delivery drivers have, in order to route them to the correct loading bay and orders. 

Assigning delivery professionals who have a bicycle to nearby orders and giving motorcyclists orders that are farther can help Scones Unlimited optimize their operations

By using AWS Sagemaker to build an image classification model that can tell bicycles apart from motorcycles

## Project Roadmap:
- Data staging
- Model training and deployment
- Deploy three Lambda functions
- Use the Step Functions visual editor to chain them together
- Testing and evaluation
## Data staging
Use a sample dataset called CIFAR
- Extract the data from a hosting service
- Transform it into a usable shape and format
- Capture all the bicycles and motorcycles and save them
  - Identify the label numbers for Bicycles and Motorcycles
  - Save all the images into the ./test and ./train directories
- Load it to S3

## Model training and deployment

- Using a AWS build-in image classification algorithm to train the model
- Create an estimator img_classifier_model
- Set a few key hyperparameters and define the inputs for the model
- Train the model by calling the fit method 
- Configure Model Monitor to track our deployment,using DataCaptureConfig
- Deploy the model with the data capture config to an endpoint
- Instantiate a Predictor

## AWS Lambda functions
Write and deploy three Lambda functions
- First lambda is responsible for data generation :
    - Get the s3 address from the Step Function event input
    - Download the data from s3
    - Encode it
    - Return it to the step function as image_data
- Second lambda is responsible for image classification :
    - Take the image output from the first lambda function
    - Decode the image data
    - Instantiate a Predictor
    - Make a prediction
    - Pass the prediction back to the the Step Function
- Third lambda is responsible for filtering out low-confidence of predictor:
    - Define a threshold between 1.00 and 0.000 for the model
    - Grab the inferences from the event
    - Check if any values in the prediction are above THRESHOLD
    - If the threshold is met :
        - Pass the data back out of the Step Function
    - Else :
        - End the Step Function with an error
## AWS Step Functions visual editor
Chain three Lambda functions together
- Construct a workflow that chains them together
- Add three of them chained together in the proper order
![stepfunctions_graph](https://github.com/lenaasayed/AWS-Machine-Learning-Engineer/assets/42747018/9921f9ce-ebcc-486e-965e-cbb1c4c4b919)

## Testing and evaluation

- Perform several step function invokations using data from the test dataset
- Create function to generate test cases inputs for invokations
- Make several executions in Step Function dashboard
- Visualize the record of predictions
