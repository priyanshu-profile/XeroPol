import os
import pickle
from utils import load_tokenizer

if __name__ == "__main__":

    do_preprocess_emowoz = True
    do_preprocess_multidogo = False

    if do_preprocess_emowoz:
        # ---------- EmoWOZ DATA PREPARATION -----------
        unique_politelabels = set()  
        unique_sentimentlabels = set()        
        task, data = 'emowoz', 'data'
        
        tokenizer = load_tokenizer('xlm-roberta-base')
        for lang in ['en', 'hi']:
            for split in ['train', 'dev', 'test']:
                print("Processing the %s split of language: %s." % (split, lang))
                if not os.path.exists(os.path.join(data, task, lang, split)):
                    os.mkdir(os.path.join(data, task, lang, split))
                all_inputs, all_polite_labels, all_sentiment_labels = [], [], []
                lines = open(os.path.join(data, task, lang, "%s.txt" % split), "r", encoding="utf-8").readlines()
                i = 0

                while i < len(lines):
                    inputs = []
                    line = lines[i].split("|")
                    if len(line) != 8:
                        print(line,split)
                    assert len(line) == 8
                    polite = line[4].strip()
                    polite_label = polite
                    sentiment = line[5].strip()
                    sentiment_label = sentiment
                    unique_politelabels.add(polite)
                    unique_sentimentlabels.add(sentiment)
                    all_tokens = eval(line[7])
                    tokens = all_tokens['tokens']
                    
                    for token in tokens:
                        tokenized = [t for t in tokenizer.tokenize(token) if t != '▁']
                        for idx, t in enumerate(tokenized):
                            inputs.append(t)
                    i += 1
                    all_inputs.append(inputs)
                    all_polite_labels.append(polite_label)
                    all_sentiment_labels.append(sentiment_label)
                dump = {'inputs': all_inputs, 'polite_labels': all_polite_labels,'sentiment_labels': all_sentiment_labels}
                pickle.dump(dump, open(os.path.join(data, task, lang, split, "data.pkl"), "wb"))
        unique_politelabels_list = list(unique_politelabels).copy()
        unique_sentimentlabels_list = list(unique_sentimentlabels).copy()
        pickle.dump(sorted(unique_politelabels_list), open(os.path.join(data, task, 'polite_labels.pkl'), 'wb'))
        pickle.dump(sorted(unique_sentimentlabels_list), open(os.path.join(data, task, 'sentiment_labels.pkl'), 'wb'))
        print("Finished.")

if do_preprocess_multidogo:
        # ---------- MultiDoGo DATA PREPARATION -----------
        unique_politelabels = set()  
        unique_sentimentlabels = set()        
        task, data = 'multidogo', 'data'
        
        tokenizer = load_tokenizer('xlm-roberta-base')
        
        for lang in ['en', 'hi']:
                for split in ['train', 'dev', 'test']:
                    print("Processing the %s split of language: %s." % (split, lang))
                    if not os.path.exists(os.path.join(data, task, lang, split)):
                        os.mkdir(os.path.join(data, task, lang, split))
                    all_inputs, all_polite_labels, all_sentiment_labels = [], [], []
                    lines = open(os.path.join(data, task, lang, "%s.txt" % split), "r", encoding="utf-8").readlines()
                    i = 0

                    while i < len(lines):
                        inputs = []
                        line = lines[i].split("|")
                        if len(line) != 5:
                            print(line)
                            print(len(line))
                            print("Processing the %s split of language: %s." % (split, lang))
                        assert len(line) == 5
                        polite = line[1].strip()
                        polite_label = polite
                        sentiment = line[2].strip()
                        sentiment_label = sentiment
                        unique_politelabels.add(polite)
                        unique_sentimentlabels.add(sentiment)
                        all_tokens = eval(line[4])
                        tokens = all_tokens['tokens']
                        
                        for token in tokens:
                            tokenized = [t for t in tokenizer.tokenize(token) if t != '▁']
                            for idx, t in enumerate(tokenized):
                                inputs.append(t)
                        i += 1
                        all_inputs.append(inputs)
                        all_polite_labels.append(polite_label)
                        all_sentiment_labels.append(sentiment_label)
                    dump = {'inputs': all_inputs, 'polite_labels': all_polite_labels,'sentiment_labels': all_sentiment_labels}
                    pickle.dump(dump, open(os.path.join(data, task+'-'+d, lang, split, "data.pkl"), "wb"))
            
            unique_politelabels_list = list(unique_politelabels).copy()
            unique_sentimentlabels_list = list(unique_sentimentlabels).copy()
            pickle.dump(sorted(unique_politelabels_list), open(os.path.join(data, task, 'polite_labels.pkl'), 'wb'))
            pickle.dump(sorted(unique_sentimentlabels_list), open(os.path.join(data, task, 'sentiment_labels.pkl'), 'wb'))
        print("Finished.")
