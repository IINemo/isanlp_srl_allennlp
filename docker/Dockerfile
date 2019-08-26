FROM inemo/isanlp_base_cuda

# Install common
# RUN apt-get update

# Install python packages
RUN pip install --upgrade allennlp
RUN python -m spacy download en_core_web_sm

COPY isanlp_srl_allennlp /src/custom_modules/isanlp_srl_allennlp

ENV PYTHONPATH=/src/custom_modules

COPY ./bert-base-srl-2019.06.17.tar.gz /src/

COPY entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]


CMD [ "python", "/start.py", "-m", "isanlp_srl_allennlp.pipeline_srl", "-a", "create_pipeline", "--no_multiprocessing", "True"]
