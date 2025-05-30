import boto3
import time

def lambda_handler(event, context):
    sm = boto3.client('sagemaker')

    job_name = f'predict-processing-job-{int(time.time())}'
    
    response = sm.create_processing_job(
        ProcessingJobName=job_name,
        RoleArn='arn:aws:iam::590183841012:role/LabRole',  # ✅ เปลี่ยนเป็น Role ของคุณ

        AppSpecification={
            'ImageUri': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1',
            'ContainerEntrypoint': ['python3'],
            'ContainerArguments': [
                '/opt/ml/processing/input/code/predict-script-python.py'
            ]
        },

        ProcessingInputs=[
            {
                'InputName': 'script-code',
                'S3Input': {
                    'S3Uri': 's3://stress-prediction-project/predict-script/',
                    'LocalPath': '/opt/ml/processing/input/code/',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File'
                }
            },
            {
                'InputName': 'input-data',
                'S3Input': {
                    'S3Uri': 's3://stress-prediction-project/combined_data_steaming/',
                    'LocalPath': '/opt/ml/processing/input/data/',
                    'S3DataType': 'S3Prefix',
                    'S3InputMode': 'File'
                }
            }
        ],

        ProcessingOutputConfig={
            'Outputs': [
                {
                    'OutputName': 'predicted-output',
                    'S3Output': {
                        'S3Uri': 's3://stress-prediction-project/prediction-results/',
                        'LocalPath': '/opt/ml/processing/output/',
                        'S3UploadMode': 'EndOfJob'
                    }
                }
            ]
        },

        ProcessingResources={
            'ClusterConfig': {
                'InstanceCount': 1,
                'InstanceType': 'ml.m5.large',
                'VolumeSizeInGB': 10
            }
        },

        StoppingCondition={
            'MaxRuntimeInSeconds': 3600
        }
    )

    return {
        'statusCode': 200,
        'body': f"Started Processing Job: {job_name}"
    }
