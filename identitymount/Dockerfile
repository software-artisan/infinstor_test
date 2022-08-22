FROM tensorflow/tensorflow
 
RUN apt update 
RUN apt install -y libfuse-dev

RUN pip install boto3
RUN pip install mlflow

RUN pip install http://vmi504080.contaboserver.net:9876/packages/parallels-plugin/parallels_plugin-0.3.3-py3-none-any.whl
