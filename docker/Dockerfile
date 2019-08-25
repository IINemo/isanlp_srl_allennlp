FROM windj007/jupyter-keras-tools

# Install common
#RUN apt-get update && apt-get install -y default-jre default-jdk
RUN apt-get update

RUN pip install --upgrade pip
RUN pip install --upgrade torch
RUN pip install --upgrade allennlp

COPY isanlp_srl_allennlp /src/custom_modules/isanlp_srl_allennlp

COPY start.py /

ENV PYTHONPATH=/src/custom_modules

COPY ./bert-base-srl-2019.06.17.tar.gz /src/

COPY entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]


CMD [ "python", "/start.py", "-m", "isanlp_srl_allennlp.pipeline_srl", "-a", "create_pipeline" ]
