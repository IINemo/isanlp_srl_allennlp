from isanlp.annotation_repr import CSentence
from isanlp.annotation import Event, TaggedSpan, Sentence, Token
from allennlp.predictors.predictor import Predictor


class ProcessorSrlAllennlp:
    def __init__(self, model_path):
        self._predictor = Predictor.from_path(model_path)
    
    def _find_object_start(self, seq, start=0):
        def is_start(idx):
            parts = seq[idx].split('-', maxsplit=1)
            if len(parts) < 2:
                return False
            
            label, _ = parts
            return label == 'B'
        
        i = start
        while i < len(seq) and not is_start(i):
            i += 1
        
        if i >= len(seq):
            return -1, ''
        
        return i, seq[i].split('-')[1]
        
    def _find_object_end(self, seq, obj_start_idx):
        i = obj_start_idx + 1
        while i < len(seq) and not seq[i].startswith('B-') and seq[i] != 'O':
            i += 1
        
        return i
        
    def _convert_format(self, allennlp_srl):
        events = []
        for verb in allennlp_srl['verbs']:
            event = Event(pred=None, args=[])
            
            i = 0
            while True:
                start, tp = self._find_object_start(verb['tags'], start=i)
                if start == -1:
                    break
                
                finish = self._find_object_end(verb['tags'], start)
                if tp == 'V':
                    event.pred = (start, finish)
                else:
                    arg = TaggedSpan(tag=tp, begin=start, end=finish)
                    event.args.append(arg)
                
                i = finish
            
            events.append(event)
        
        return events      
    
    def __call__(self, tokens, sentences):
        result = []
        for sent in sentences:
            sent_repr = CSentence(tokens, sent)
            allennlp_srl = self._predictor.predict_tokenized(tokenized_sentence=[e.text for e in sent_repr])
            events = self._convert_format(allennlp_srl)
            result.append(events)
        
        return result
    