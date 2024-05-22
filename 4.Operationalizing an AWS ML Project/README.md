
# Operationalizing an AWS ML Project

Train and deploy an image classification model on AWS Sagemaker.using several important tools and features of AWS to adjust, improve, configure, and prepare the model for production-grade deployment.




## Project Roadmap:
- Data Preparation
- Model Training and Tuning
- Model Profiling and Debugging
- Model Deployment and Testing
- EC2 Training
- Setting up a Lambda function
- Security and testing
## Data Preparation
Process and prepare image data for training

- Download dataset
- Unzip files
- Upload the data to S3 bucket so that SageMaker can use it for training

## Model Training and Tuning

Train model using SageMaker and fine-tune it with hyperparameter optimization
- Hyperparameter Tuning
    - Finetune a pretrained model with hyperparameter tuning
    - Declare hyperparameter ranges, metrics
    - Create estimators for hyperparameters
    - Train the estimator
    - Get the best estimators and the best hyperparameters

## Model Profiling and Debugging

- Amazon SageMaker Debugger
    - Amazon SageMaker Debugger is a capability of SageMaker that provides    tools to register hooks to callbacks to extract model output tensors and save them in Amazon Simple Storage Service. 
    - It provides built-in rules for detecting model convergence issues, such as overfitting, saturated activation functions, vanishing gradients, and more. 
    - You can also set up the built-in rules with Amazon CloudWatch Events and AWS Lambda for taking automated actions against detected issues

- Amazon SageMaker Profiler

    - Amazon SageMaker Profiler is a profiling capability of SageMaker with which you can deep dive into compute resources provisioned while training deep learning models, and gain visibility into operation-level details. 

    - With SageMaker Profiler, you can track all activities on CPUs and GPUs, such as CPU and GPU utilizations, kernel runs on GPUs, kernel launches on CPUs, sync operations, memory operations across CPUs and GPUs, latencies between kernel launches and corresponding runs, and data transfer between CPUs and GPUs.

    - SageMaker Profiler also offers a user interface (UI) that visualizes the profile, a statistical summary of profiled events, and the timeline of a training job for tracking and understanding the time relationship of the events between GPUs and CPUs.

## Model Deployment and Testing
- Deploy the trained model to an endpoint 
- Test it with real-world data
- Run an prediction on the endpoint

## EC2 Training
- create an EC2 Spot instance as it is very important to minimize cost
- Spot Instance
    - AWS has a huge supply of computing resources that it makes available to users. Sometimes, people turn on instances and then leave them idle.
    - AWS makes use of these resources: as long as the person who reserved them isn’t using them, AWS makes them available temporarily to others.so we used them by using Spot Instance
- Preparing for EC2 model training
- Training and saving on EC2

## Setting up a Lambda function

 Lambda functions enable the model to be accessed by API's and other programs
- Change the name of endpoint 
- Take the input as the json file
- Decode the image file
- Get the result of the classifier 
- Return the output of the prediction

## Security and testing

- Lambda function will only be able to invoke endpoint if it has the proper security policies attached to it.
- Test the lambda function

## Reference

[Amazon SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debug-and-improve-model-performance.html.)

[Amazon SageMaker Profiler](https://docs.aws.amazon.com/sagemaker/latest/dg/train-profile-computational-performance.html.)

[Amazon Spot Instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html.)
