import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from transformers import BertTokenizer


csvfile = "data/spam.csv"
nltk.download("stopwords")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

try: 
    stopwords = stopwords.words("english")
except: 
    nltk.download("stopwords")
    stopwords = stopwords.words("english")

try:
    nlp = spacy.load("en_core_web_sm")
except: 
    spacy.cli.download("en")
    nlp = spacy.load("en_core_web_sm")

def get_x(csvfile=csvfile) -> list[str]: 
    df = pd.read_csv(csvfile, encoding="ISO-8859-1")
    x = df["v2"]
    return x

def get_y(csvfile=csvfile) -> list[int]: 
    df = pd.read_csv(csvfile, encoding="ISO-8859-1")
    y = df["v1"]
    nums = []
    for element in y: 
        if element == "ham": nums.append(float(0))
        else: nums.append(float(1))
    return nums

def tokenize_x(x: list[str]) -> list[list[float]]: 
    dictionary = {}
    count = 0
    tokenized = []
    for text in x: 
        words = lemmatize_text(text)
        for i, word in enumerate(words): 
            if word not in dictionary: 
                dictionary[word] = count
                words[i] = count
                count += 1
            else: 
                words[i] = dictionary[word]
        tokenized.append(words)
    return tokenized

def bert_tokenize(x: list[str]) -> list[list[int]]: 
    tokenized_x = []
    for text in x: 
        text = tokenizer.encode(text, 
                         add_special_tokens=True,
                         max_length=25, 
                         truncation=True, 
                         padding='max_length'
                         )
        tokenized_x.append(text)
    return tokenized_x


def lemmatize_text(text: str, padding=25) -> list[str]: 
    # lemmatization and removing stop words
    spacy_text = nlp(text.lower())
    words = [word.lemma_ for word in spacy_text if word not in stopwords]
    padded = words[:padding] + [0] * (padding - len(words))
    return padded
