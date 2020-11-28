from spacy.lang.en import English
import spacy


"""
Guide https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/
"""


class Tokenizer(object):
    # English
    nlp = English()
    # Add pipeline 'sentencizer' component - break text into sentences
    #nlp.add_pipe(nlp.create_pipe('sentencizer'))

    def __init__(self):
        pass

    @classmethod
    def tokenize_text(cls, text: str):
        # initial tokenize
        doc = cls.nlp(text)
        toks = [tk for tk in doc]

        # remove punctuations
        toks = [tk for tk in toks if not tk.is_punct]

        # debug
        #for tk in doc:
        #    print(tk.text, ': ', tk.lemma_, tk.pos_, tk.is_punct)

        # Removing stop words
        #toks = [tk for tk in toks if not tk.is_stop]

        # 'PRON' means pronoun
        ws = [tk.text if tk.pos_ == 'PRON' else tk.lemma_ for tk in toks]
        # lowercase all
        ws = [w.lower() for w in ws]
        return ' '.join(ws).strip()


if __name__ == '__main__':
    text = """I'm telling you this, when learning data science, you shouldn't get discouraged! Challenges and setbacks aren't failures, they're just part of the journey. You've got this!"""
    #text = """Trump is on the move for China. Obama's care is to be continued!"""
    print(Tokenizer.tokenize_text(text))
