import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en")
db = DocBin().from_disk("misc/training_data.spacy")
for doc in db.get_docs(nlp.vocab):
    assert doc.has_annotation("TAG")
    # or just inspect the tags
    for token in doc:
        print(token.text, token.tag_)