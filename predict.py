import collections
import csv
import os
import modeling
import optimization_finetuning as optimization
import tokenization
import tensorflow as tf
import pickle
import tf_metrics
import requests
import json
import re

tf_serving_ner_url = 'http://localhost:8611/v1/models/ner:predict'

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label



class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    # self.is_real_example = is_real_example


def write_tokens(tokens,mode):
  if mode=="test":
    path = os.path.join('tokens', "token_"+mode+".txt")
    wf = open(path,'a')
    for token in tokens:
      if token!="**NULL**":
        wf.write(token+'\n')
    wf.close()


def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, mode = None):
  textlist = list(example.text)
  tokens = []
  # print(textlist)
  for i, word in enumerate(textlist):
      token = tokenizer.tokenize(word)
      tokens.extend(token)
  if len(tokens) >= max_seq_length - 1:
      tokens = tokens[0:(max_seq_length - 2)]
  ntokens = []
  segment_ids = []
  label_ids = [0] * max_seq_length
  ntokens.append("[CLS]")
  segment_ids.append(0)
  for i, token in enumerate(tokens):
      ntokens.append(token)
      segment_ids.append(0)
  ntokens.append("[SEP]")
  segment_ids.append(0)
  # append("O") or append("[SEP]") not sure!
  input_ids = tokenizer.convert_tokens_to_ids(ntokens)
  input_mask = [1] * len(input_ids)
  #label_mask = [1] * len(input_ids)
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    # we don't concerned about it!
    ntokens.append("**NULL**")
    #label_mask.append(0)
  # print(len(input_ids))
  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
      [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

  feature = InputFeatures(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    label_ids=label_ids,
    #label_mask = label_mask
  )
  write_tokens(ntokens,mode)
  return feature


def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label2id, max_seq_length, tokenizer, 'infer')

    features.append(feature)
  return features


def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


with open('albert_base_ner_checkpoints/label2id.pkl','rb') as rf:
  label2id = pickle.load(rf)
  id2label = {value:key for key,value in label2id.items()}


def get_sentence_examples(lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text = tokenization.convert_to_unicode(line)
      label = tokenization.convert_to_unicode('0'*len(text))
      examples.append(InputExample(guid=guid, text=text, label=label))
    return examples

def get_labels():
    # return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]","[SEP]"]
    return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]


def get_entity(tag_seq, char_seq):
    PER = get_PER_entity(tag_seq, char_seq)
    LOC = get_LOC_entity(tag_seq, char_seq)
    ORG = get_ORG_entity(tag_seq, char_seq)
    return PER, LOC, ORG


def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append(per)
                del per
            per = char
            if i+1 == length:
                PER.append(per)
        if tag == 'I-PER':
            if 'per' not in locals().keys():
                org = char
            else:
                per += char
            if i+1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            loc = char
            if i+1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            if 'loc' not in locals().keys():
                org = char
            else:
                loc += char
            if i+1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            org = char
            if i+1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            if 'org' not in locals().keys():
                org = char
            else:
                org += char
            if i+1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG


def get_inputs_hard_clip(sent, max_seq_length):
    max_seq_length = max_seq_length - 2 # 去除开始和结尾符实际的字符大小
    sent = re.sub(r'[^\u4e00-\u9fa5,，.。;；!?]', '', sent.strip())
    if len(sent) <= max_seq_length:
        return [sent]
    sents = []
    start = 0
    end = max_seq_length
    while end > start:
        sents.append(sent[start:end])
        start = end
        end = len(sent) if end + max_seq_length > len(sent) else end + max_seq_length
    return sents


def get_ner(sent, max_seq_length):
    if sent == '' or sent.isspace():
        return None
    else:
        sents = get_inputs_hard_clip(sent, max_seq_length)
        predict_examples = get_sentence_examples(sents, 'infer')
        features = convert_examples_to_features(predict_examples, label2id, max_seq_length, tokenizer)
        feed_dict = []
        for i in range(0, len(features)):
            feed_dict.append({"input_ids": features[i].input_ids,
                          "input_mask": features[i].input_mask,
                          "segment_ids": features[i].segment_ids,
                          "label_ids": features[i].label_ids
                          })

        data = json.dumps({"instances": feed_dict})
        headers = {"content-type": "application/json"}
        print(feed_dict)
        # curl http://localhost:8611/v1/models/ner
        json_response = requests.post(tf_serving_ner_url, data=data, headers=headers)
        print(json_response.text)
        predictions = json.loads(json_response.text)['predictions']
        print(predictions)
        pers, locs, orgs = [], [], []
        for i in range(0, len(predictions)):
            tag = [id2label[id] for id in predictions[i] if id != 0]
            output_line = " ".join(tag) + "\n"
            print(output_line)
            per, loc, org = get_entity(tag[1:-1], list(sents[i]))       # tag 要去掉开始和结束标识
            pers += per
            locs += loc
            orgs += org
        # PER = list(set([p.strip() for p in PER if re.match(PT_KW_WHITE, p.strip())]))
        # LOC = list(set([p.strip() for p in LOC if re.match(PT_KW_WHITE, p.strip())]))
        # ORG = list(set([p.strip() for p in ORG if re.match(PT_KW_WHITE, p.strip())]))
        print('final result:', str(pers), str(locs), str(orgs))

        return pers, locs, orgs


max_seq_length=512
vocab_file='./albert_config/vocab.txt'
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
sentence = '这是一个测试，目的是识别北京'

get_ner(sentence, max_seq_length)



