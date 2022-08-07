FROM tensorflow/tensorflow
 
RUN apt update 
RUN apt install -y libfuse-dev

RUN pip install boto3
RUN pip install mlflow

RUN pip install http://vmi504080.contaboserver.net:9876/packages/infinstor-mlflow-plugin/infinstor_mlflow_plugin-2.0.41-py3-none-any.whl
RUN pip install http://vmi504080.contaboserver.net:9876/packages/clientlib/infinstor-2.0.37-py3-none-any.whl
