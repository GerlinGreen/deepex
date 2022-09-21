import spacy

from .generator_utils import WhitespaceTokenizer, span_filter_func, get_empty_candidates
from ..utils import *
from transformers import pipeline


class NPMentionGenerator:

    def __init__(self):
      # if nlp == "ckiplab/bert-base-han-chinese" \
      #   or  nlp == "bert-base-chinese" :
      #   self.nlp = 'zh_core_web_sm'
      # else :
      #   self.nlp = 'en_core_web_sm'
      # spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'textcat'])
      # self.tokenizer = spacy.load('en_core_web_sm')
      # self.whitespace_tokenizer = spacy.load('en_core_web_sm')
      #------------------------------Spacy ZH----------------------------------------
      spacy.load('zh_core_web_sm', disable=['tagger', 'parser', 'textcat'])
      self.tokenizer = spacy.load('zh_core_web_sm')
      self.whitespace_tokenizer = spacy.load('zh_core_web_sm')
      self.whitespace_tokenizer.tokenizer = WhitespaceTokenizer(self.whitespace_tokenizer.vocab)
      #------------------------------Spacy ZH----------------------------------------

    def get_mentions_raw_text(self, text: str, whitespace_tokenize=False, extra=None):
        
        #------------------------------CKIP Han----------------------------------------
        task = "token-classification"
        model_name="ckiplab/bert-base-chinese-pos"
        classifier = pipeline(task, model=model_name)
        #------------------------------CKIP Han----------------------------------------
        
        #------------------------------Spacy ZH----------------------------------------
        if whitespace_tokenize:
            tokens = self.whitespace_tokenizer(text)
        else:
            #------------------------------Spacy ZH----------------------------------------
            # self.tokenizer.max_length = 1000000000
            # tokens = self.tokenizer(text)
            #------------------------------Spacy ZH----------------------------------------
            #------------------------------CKIP Han----------------------------------------
            tokens = classifier(text)
            #------------------------------CKIP Han----------------------------------------
        
        #------------------------------Spacy ZH----------------------------------------
        # _tokens = [t.text for t in tokens]
        #------------------------------Spacy ZH----------------------------------------
        #------------------------------CKIP Han----------------------------------------
        _tokens = [t["word"] for t in tokens]
        #------------------------------CKIP Han----------------------------------------
        spans_to_candidates = {}
        spans_to_positions = {}
        zh_chunks = []
        
        #------------------------------Spacy ZH----------------------------------------
        # # parts of speech tagging
        # PRENOUN = -1
        # char_num = 0
        # for t, token in enumerate(tokens):
        #     if token.pos_ == 'NOUN' :
        #         if PRENOUN == t-1 and len(zh_chunks)>0:
        #             try :
        #                 zh_chunks[-1][0] = zh_chunks[-1][0] + token.text
        #                 zh_chunks[-1][2] = t + 1
        #                 zh_chunks[-1][4] = zh_chunks[-1][4] + len(token)
        #             except :
        #                 raise AssertionError("zh_chunks: ", zh_chunks)
        #         else :
        #             zh_chunks.append([token.text, t, t + 1, char_num, char_num + len(token)])
        #         PRENOUN = t

        #     char_num = char_num + len(token)
        # print(zh_chunks)
        #------------------------------Spacy ZH----------------------------------------
        
        #------------------------------CKIP Han----------------------------------------
        token_list = []
        tokens_dict = {"sent": text}
        pre_token = {'entity': '', 
                'score': 0,
                'index': 0, 
                'word': '',
                'start': 0,
                'end': 0}
        tmp_token = ""

        for t, token in enumerate(tokens) :
          tokens_dict[t] = token
          if pre_token["entity"] != "" :
            if token["entity"] == pre_token["entity"] :
              tmp_token = tmp_token + token["word"]
            else :
              token_list.append(tmp_token)
              tmp_token = token["word"]
          else :
            tmp_token = token["word"]
          pre_token = token

        zh_chunks = []
        PRENOUN = -1
        char_num = 0
        for t, token in enumerate(tokens_dict) :
          if token != 'sent' :
            # print(tokens_dict[token]['entity'])
            if tokens_dict[token]['entity'] == 'Na' or tokens_dict[token]['entity'] == 'Nb' :
                if PRENOUN == t-1 and len(zh_chunks)>0 :
                    zh_chunks[-1][0] = zh_chunks[-1][0] + tokens_dict[token]["word"]
                    zh_chunks[-1][2] = t + 1
                    zh_chunks[-1][4] = zh_chunks[-1][4] + len(tokens_dict[token]["word"])
                else :
                    zh_chunks.append([tokens_dict[token]["word"], t, t + 1, char_num, char_num + len(tokens_dict[token]["word"])])
                PRENOUN = t

          char_num = char_num + len(tokens_dict[token])
        print(zh_chunks)
        #------------------------------CKIP Han----------------------------------------

        for cand in zh_chunks:
            spans_to_candidates[(cand[1], cand[2]-1)] = [(None, cand[0], 1.0)]
            spans_to_positions[(cand[1], cand[2]-1)] = [cand[3], cand[4]]


        # for cand in tokens.noun_chunks:
        #     spans_to_candidates[(cand.start, cand.end-1)] = [(None, cand.text, 1.0)]
        #     spans_to_positions[(cand.start, cand.end-1)] = [cand.start_char, cand.end_char]

        spans = []
        entities = []
        priors = []
        positions = []
        for span, candidates in spans_to_candidates.items():
            spans.append(list(span))
            entities.append([x[1] for x in candidates])
            mention_priors = [x[2] for x in candidates]

            sum_priors = sum(mention_priors)
            priors.append([x/sum_priors for x in mention_priors])

            positions.append(spans_to_positions[span])
        ret = {
            "tokenized_text": _tokens,
            "candidate_spans": spans,
            "candidate_entities": entities,
            "candidate_entity_priors": priors,
            "candidate_positions": positions,
            
            "head_candidate_spans": [],
            "head_candidate_entities": [],
            "head_candidate_entity_priors": [],
            "head_candidate_positions": [],
            
            "tail_candidate_spans": [],
            "tail_candidate_entities": [],
            "tail_candidate_entity_priors": [],
            "tail_candidate_positions": [],
            
            "relation_candidate_spans": [],
            "relation_candidate_entities": [],
            "relation_candidate_entity_priors": [],
            "relation_candidate_positions": [],
        }

        if len(spans) == 0:
            ret.update(get_empty_candidates())

        return ret
