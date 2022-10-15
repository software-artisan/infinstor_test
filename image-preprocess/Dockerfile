FROM tensorflow/tensorflow
 
RUN pip install boto3
RUN pip install mlflow
RUN pip install tensorflow
RUN pip install tensorflow_datasets
RUN pip install tensorflow_addons
RUN pip install pandas
RUN pip install numpy
RUN pip install pillow

ARG IGNORECACHE=4
RUN echo "$IGNORECACHE"
RUN python3 -m pip install --no-cache-dir http://vmi504080.contaboserver.net:9876/packages/concurrent-plugin/concurrent_plugin-0.3.11-py3-none-any.whl
RUN python3 -m pip install --no-cache-dir http://vmi504080.contaboserver.net:9876/packages/clientlib/infinstor-2.0.46-py3-none-any.whl
