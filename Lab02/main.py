# Segmentation
def task_one():
    import nltk
    from nltk.tokenize import sent_tokenize
    # Download the tokenizer models (only once)
    nltk.download('punkt')
    # Sample paragraph
    text = "Away from the city lights, we were able to see the Milky Way."
    # Split into sentences
    sentences = sent_tokenize(text)
    # Show the segmented sentences
    print("Segmented Sentences:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")

# Tokenization
def task_two():
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Download required data
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    # Example sentence
    sentence = "The runners were running and caring about their connections."
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Apply stemming
    stemmed = [stemmer.stem(word) for word in tokens]
    # Apply lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    # Print results
    print("Original :", tokens)
    print("Stemmed :", stemmed)
    print("Lemmatized :", lemmatized)

# Removing stop words
def task_three():
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    # Download stopwords (only needed once)
    nltk.download("punkt")
    nltk.download("stopwords")
    # Sample sentence
    text = "Away from the city lights, we were able to see the Milky Way"
    # Step 1: Tokenize the sentence
    tokens = word_tokenize(text)
    # Step 2: Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    # Print results
    print("Original tokens:", tokens)
    print("Filtered tokens (no stop words):", filtered_tokens)

# Stemming
def task_four():
    import nltk
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    # Download necessary data if running for the first time
    nltk.download("punkt")
    # Create stemmer
    stemmer = PorterStemmer()
    # Example sentence
    sentence = "we were able to see the Milky Way"
    # Tokenize sentence
    tokens = word_tokenize(sentence)
    # Stem each word
    stemmed_words = [stemmer.stem(word) for word in tokens]
    # Show results
    print("Original words:", tokens)
    print("Stemmed words :", stemmed_words)

# Lemmatization
def task_five():
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Download required data
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    # Example sentence
    sentence = "we were able to see the Milky Way"
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Apply stemming
    stemmed = [stemmer.stem(word) for word in tokens]
    # Apply lemmatization
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    # Print results
    print("Original :", tokens)
    print("Stemmed :", stemmed)
    print("Lemmatized :", lemmatized)

# POS Tagging
def task_six():
    import nltk
    from nltk import pos_tag, word_tokenize
    # Download necessary data (only the first time)
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    # Sample sentence
    sentence = "Wow! The excited dog quickly ran across the yard because it saw another cat."
    # Step 1: Tokenize the sentence
    tokens = word_tokenize(sentence)
    # Step 2: POS tagging
    tagged = pos_tag(tokens)
    # Output results
    print("POS Tagging Results:")
    for word, tag in tagged:
        print(f"{word:10} -> {tag}")

# Named Entity Recognition
def task_seven():
    # No Code Provided
    ...

# Custom Text Preprocessing
def task_eight():
    import string

    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    # Download necessary NLTK data (only first time)
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    import nltk
    nltk.download('averaged_perceptron_tagger_eng')
    # Helper function to map NLTK POS tags to WordNet POS tags
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN # default to noun
    # Tokenize function
    def tokenize(text, remove_stopwords=True, normalize='lemmatize'):
        # Lowercase
        text = text.lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Remove punctuation
        tokens = [w for word in tokens for w in word.strip(string.punctuation).split()]
        # Remove stopwords
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [w for w in tokens if w not in stop_words]
        # Lemmatize or stem
        if normalize == 'lemmatize':
            lemmatizer = WordNetLemmatizer()
            tagged_tokens = nltk.pos_tag(tokens)
            tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in tagged_tokens]
            return tokens

    # Test sentence
    sentence = "The children were running faster and performed better while studying different programming languages"

    cleaned_sentence = tokenize(sentence, remove_stopwords=True, normalize='lemmatize')

    print(cleaned_sentence)

if __name__ == "__main__":
    task_one()

    task_two()

    task_three()

    task_four()

    task_five()

    task_six()

    task_seven() # This will do nothing

    task_eight()