FROM index.caicloud.io/caicloud/ml-libs

RUN apt-get update && apt-get install -y bc

RUN rm -rf /notebooks/*

COPY data /tmp/data
COPY file_server.py /file_server.py
COPY run_tf.sh /run_tf.sh

COPY notebooks /notebooks
COPY distributed /distributed

CMD ["/run_tf.sh"]
