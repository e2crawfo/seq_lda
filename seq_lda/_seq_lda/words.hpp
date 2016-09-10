#pragma once

#include <vector>
#include <memory>

using namespace std;

typedef vector<int> Word;

class Document {
public:
    Document() = default;
    Document(int n_word_types):n_word_tokens(0), n_word_types(0){
        word_indices.reserve(n_word_types);
        word_counts.reserve(n_word_types);
    };

    void add_word(int word_idx, int word_count){
        word_indices.push_back(word_idx);
        word_counts.push_back(word_count);
        n_word_tokens += word_count;
        n_word_types++;
    }

    int n_word_tokens;
    int n_word_types;
    vector<int> word_indices;
    vector<int> word_counts;
};

class Dictionary {
public:
    Dictionary() = default;
    Dictionary(int n_word_types, int n_symbols)
    :n_word_types(n_word_types), n_symbols(n_symbols){
        words.reserve(n_word_types);
    }

    void add_word(Word& word){
        words.push_back(word);
    }

    void add_word(Word&& word){
        words.push_back(move(word));
    }

    const vector<Word>& get_words() const{
        return words;
    }

    vector<Word> words;
    int n_word_types;
    int n_symbols;
};

class Corpus {
public:
    Corpus() = default;
    Corpus(int n_docs, int n_word_types, int n_symbols)
    :n_docs(n_docs), n_word_types(n_word_types), n_symbols(n_symbols){
        docs.reserve(n_docs);
    }

    void add_doc(Document& doc){
        docs.push_back(doc);
    }

    void add_doc(Document&& doc){
        docs.push_back(move(doc));
    }

    void set_dict(const Dictionary& _dictionary){
        dictionary = _dictionary;
    }

    void set_dict(Dictionary&& _dictionary){
        dictionary = move(_dictionary);
    }

    int max_doc_length() const{
        int max = 0;
        for(auto& doc: docs){
            if(doc.n_word_types > max){
                max = doc.n_word_types;
            }
        }
        return(max);
    }

    int n_docs;
    int n_word_types;
    int n_symbols;

    Dictionary dictionary;
    vector<Document> docs;
};
