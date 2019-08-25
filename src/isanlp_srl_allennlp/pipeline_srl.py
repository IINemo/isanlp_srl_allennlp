from isanlp.processor_sentence_splitter import ProcessorSentenceSplitter
from isanlp.en.processor_tokenizer_nltk_en import ProcessorTokenizerNltkEn
from isanlp import PipelineCommon

from .processor_srl_allennlp import ProcessorSrlAllennlp


def create_pipeline(delay_init):
    model_path = '/src/bert-base-srl-2019.06.17.tar.gz'
    
    tokenizer = ProcessorTokenizerNltkEn()
    splitter = ProcessorSentenceSplitter()
    srl_proc = ProcessorSrlAllennlp(model_path)

    pipeline_default = PipelineCommon([
        (tokenizer, ['text'], {0 : 'tokens'}),
        (splitter, ['tokens'], {0 : 'sentences'}),
        (srl_proc, ['tokens', 'sentences'], {0 : 'srl'})
    ], name='default')
    
    return pipeline_default
