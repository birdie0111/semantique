import pathlib
from gensim.models import KeyedVectors as kv
import spacy
from scipy.stats import hmean
import json


# chemin vers le fichier des plongement lexicaux
embfile = "./frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin"

# Charger les plongements lexicaux en mémoire
wv = kv.load_word2vec_format(embfile, binary=True, encoding='UTF-8', unicode_errors='ignore')

# Charger spacy avec le modèle du français
spacy_nlp = spacy.load('fr_core_news_md')


# Pour chacun des trois aspects, on fournit des mots-exemples qui seront utilisés pour calculer
# des scores de similarité avec chaque token du texte afin de décider s'il exprime un des
# trois aspects
aspects = {
    'nourriture': ['dessert', 'poisson', 'riz', 'pâtes', 'purée', 'viande', 'sandwich', 'frites'],
    'boisson': ['eau', 'vin', 'limonade', 'bière', 'jus', 'thé', 'café'],
    'service': ["service", 'serveur', 'patron', 'employé'],
}


# Similarité moyenne entre un mot et un ensemble de mots : on prend la moyenne harmonique des distances
# puis on la soustrait à 1 pour obtenir la mesure inverse (la similarité), et on arrondit à 4 décimales.
def get_sim(word, other_words):
    if word not in wv.key_to_index:
        return 0
    dpos = wv.distances(word, other_words)
    d = hmean(abs(dpos))
    return round((1 - d),4)


# Pour un token spacy, cette méthode décide si c'est un terme d'aspect en cherchant l'aspect pour
# lequel il a une similarité maximale (calculée avec les mots-exemples des aspetcs).
# si le score maxi est plus petit que 0.5, il n'y pas d'aspect et la méthode retourne None
def get_aspect_emb(token):
    if token.pos_ != "NOUN":
        return None
    aspect_names = [aspect_name for aspect_name in aspects]
    scores = [(aspect_name,get_sim(token.lemma_, aspects[aspect_name])) for aspect_name in aspect_names]
    scores.sort(key=lambda x:-x[1])
    max_score = scores[0][1]
    max_aspect = scores[0][0] if max_score >= 0.5 else None
    return max_aspect


######################################################################################################
# Ne pas modifier le code ci-dessus
# Rajoutez votre code d'extractions après ce commentaire
# vous devez utiliser la méthode get_aspect_emb(token) définie ci-dessus pour savoir si un token
# est un terme d'aspect et (quel aspect)
######################################################################################################
import sys
import time
negation = ["pas", "non", "peu", "guère", "mal", "trop"]

def main():
    if (len(sys.argv) != 2):
        # Verifier si nombre d'arguments est correcte
        print("Nombre d'arguments pas correcte, il faut que utiliser le nom de texte comme argument.\n")
    else:
        results = []
        filename = sys.argv[1]
        with open(filename, "r", encoding="utf-8") as f_in:
            phrases = []
            for phrase in f_in:
                phrases.append(phrase)
                
            for doc in spacy_nlp.pipe(phrases):
                for sent in doc.sents:
                    triplets = [] # une list de triplets pour chaque phrase
                    adjs = []
                    term = ""
                    for tok in sent:
                        if (tok.pos_ == "ADJ"):
                            adjs.append(tok)
                        else:
                            if (term == ""):
                                max_aspect = get_aspect_emb(tok)
                                if (max_aspect):                        
                                # Si le premier nom dans aspect trouvé
                                    term = tok.text
                                    main_tok = tok # gardre le tok pour future vérifications
                    # Vérification des contraintes:
                    if (term != ""):
                        for adj in adjs:
                            adjectif = adj.text
                            # Condition de négation
                            if (adj.children):
                                for child in adj.children:
                                    if (child.dep_ == "advmod" and child.text in negation):
                                        adjectif = child.text + "_" + adj.text
                            # Condition 1: amod
                            if (adj.head == main_tok and adj.dep_ == "amod"):
                                #adj_head_conj.append(adj)
                                triplets.append( [max_aspect, term, adjectif] )
                            # Condition 2: nsubj, nsubj:pass
                            if (main_tok.head == adj and (main_tok.dep_ == "nsubj" or main_tok.dep_ == "nsubj:pass")):
                                #adj_head_conj.append(adj)
                                triplets.append( [max_aspect, term, adjectif] )
                            # Condition 3: a2 conj avec a1, et a1 soit dans condition 1, soit condition 2
                            if (adj.head in adjs and adj.dep_ == "conj"):
                                triplets.append( [max_aspect, term, adjectif] )
                        if (triplets != []):
                            result = {"phrase": phrase, "triplets": triplets} 
                            results.append(result)

    content = json.dumps(results, indent=4, sort_keys=True, ensure_ascii=False) 
    with open ("try.json", "w", encoding="utf-8") as f_out:
        f_out.write(content)  
                    
                            

                        
start_time = time.time()
main()
end_time = time.time()
print("The program executed: " + str(end_time - start_time))