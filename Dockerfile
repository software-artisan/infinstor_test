FROM tensorflow/tensorflow
 
RUN apt update 
RUN apt install -y libfuse-dev

RUN pip install boto3
RUN pip install mlflow

RUN pip install http://vmi504080.contaboserver.net:9876/packages/clientlib/infinstor-2.0.39-py3-none-any.whl

