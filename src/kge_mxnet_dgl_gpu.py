import sagemaker
from sagemaker.mxnet import MXNet

sagemaker_session = sagemaker.Session()

mx_estimator = MXNet(
    sagemaker_session=sagemaker_session,
    entry_point="smtrain.py",
    source_dir="../benchmarks/dgl-gpu/mx",
    role="SageMakerRole",
    train_instance_count=1,
    train_instance_type="ml.p3.2xlarge",
    image_name="841569659894.dkr.ecr.us-east-2.amazonaws.com/beta-mxnet-training:1.6.0-py3-gpu-build",
    py_version="py3",
    output_path="s3://bai-results-sagemaker/dgl",
)

mx_estimator.fit(logs=True, wait=True)
