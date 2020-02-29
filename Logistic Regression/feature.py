import sys

def read_dict(dict_input):
    corpus = {}
    with open(dict_input, 'r') as f:
        for line in f:
            line = line.split(' ')
            corpus[line[0]] = int(line[1])
    return corpus

def read_data(read_path):
    text, label = [], []
    with open(read_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            label.append(int(line[0]))
            text.append(line[1])
    return label, text

def vectorizer(text, corpus, flag, t=4):
    result = []
    if flag == 1:
        for record in text:  # each record is a x[i]
            record = record.strip().split(' ')
            current_feature = []
            sorted_unique_record = sorted(set(record), key=record.index)
            for unique_word in sorted_unique_record:
                if unique_word in corpus:
                    current_feature.append('%d:1' % corpus[unique_word])
            result.append(current_feature)
    elif flag == 2:
        for record in text:  # each record is a x[i]
            record = record.strip().split(' ')
            current_feature = []
            sorted_unique_record = sorted(set(record), key=record.index)
            for unique_word in sorted_unique_record:
                if unique_word in corpus and record.count(unique_word) < t:
                    current_feature.append('%d:1' % corpus[unique_word])
            result.append(current_feature)
    return result # a 2-d list, row is each index:count

def output(path, labels, texts):
    with open(path, 'w') as f:
        for idx, text in enumerate(texts):
            f.write(str(labels[idx])+'\t')
            for feature in text:
                f.write(feature + '\t')
            f.write('\n')

def main():
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])

    corpus = read_dict(dict_input)
    train_labels, train_texts = read_data(train_input)
    val_labels, val_texts = read_data(validation_input)
    test_labels, test_texts = read_data(test_input)

    train_formatted_texts = vectorizer(train_texts, corpus, feature_flag)
    val_formatted_texts = vectorizer(val_texts, corpus, feature_flag)
    test_formatted_texts = vectorizer(test_texts, corpus, feature_flag)

    output(formatted_train_out, train_labels, train_formatted_texts)
    output(formatted_validation_out, val_labels, val_formatted_texts)
    output(formatted_test_out, test_labels, test_formatted_texts)

if __name__ == '__main__':
    main()