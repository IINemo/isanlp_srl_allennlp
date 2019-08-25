#!/bin/bash

script_dir=$(dirname $0)
echo $script_dir

if [ ! -f "$script_dir/bert-base-srl-2019.06.17.tar.gz" ]; then
    wget -O "$script_dir/bert-base-srl-2019.06.17.tar.gz" https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz ;
else
   echo "Model file exists. Downloading is skipped."
fi

rsync -r $script_dir/../src/isanlp_srl_allennlp $script_dir
docker build -t inemo/isanlp_srl_allennlp $script_dir
