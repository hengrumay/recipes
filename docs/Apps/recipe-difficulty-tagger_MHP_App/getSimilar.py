# Press COMMAND + ENTER to run a single line in the console

# Press COMMAND + ENTER with text selected to run multiple lines
# For example, select the following lines
# and remember to press COMMAND + ENTER

# You can also run code directly in the console below.

#####################################################################################

import pandas as pd
import numpy as np

### LOAD Model Files for AWS app-path
#modelpath = '/Users/hrm/Documents/Dropbox/DSrelated/Metis/recipes/recipe-difficulty-tagger_MHP_App/models/'
#modelpath = '/var/www/FlaskApps/recipe_menuplannerApp/models/'
modelpath = '/home/ubuntu/hrm/FlaskApps/aws_recipe_menuplannerApp/models/' # EC2 oregon amazon.pem
modelpicklepath = modelpath+"bbcgfd_ilda_tf_100Ingredtopics.pkl"

### LOAD CSV File
# staticpath = '/Users/hrm/Documents/Dropbox/DSrelated/Metis/recipes/recipe-difficulty-tagger_MHP_App/static/'
# recipeListpath = staticpath + "bbcgdfood_recipelist4model_recipe-diff-tag_app.csv"
recipeListpath = modelpath + "bbcgdfood_recipelist4model_recipe-diff-tag_app.csv"

recipeDF = pd.read_csv(recipeListpath,index_col=0)

#####################################################################################

DiffN = recipeDF.reset_index().prepDifficulty.to_frame()
DiffN
DiffN[DiffN=='Easy'] = 1
DiffN[DiffN=='More effort'] = 2
DiffN[DiffN=='A challenge'] = 3
# test.prepDifficulty.unique()
recipeDF['prepDiffN'] = DiffN
# recipeDF

#####################################################################################

## Retrieve NLP / LDA Models

# load nltk's SnowballStemmer as variabled 'stemmer'
import nltk
#nltk.download() ## need to download punkt
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# define here a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    #tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


### RETRIEVE OUTPUTS
import pickle

# # ilda_tf, Ingdtm_tf, itf_vectorizer, pyldavisI
# with open('bbcgfd_ilda_tf_100Ingredtopics.pkl', 'wb') as f:
#     pickle.dump(ilda_tf, f) #model
#     pickle.dump(Ingdtm_tf, f) #doc-token-vec
#     pickle.dump(itf_vectorizer, f) #vectorizer
#     pickle.dump(pyldavisItf, f) #pyLDAvis

# # mlda_tf, Methoddtm_tf, mtf_vectorizer, pyldavisM
# with open('bbcgfd_mlda_tf_80methodtopics.pkl', 'wb') as f:
#     pickle.dump(mlda_tf, f)
#     pickle.dump(Methoddtm_tf, f)
#     pickle.dump(mtf_vectorizer, f)
#     pickle.dump(pyldavisMtf, f)

    
with open(modelpicklepath, 'rb') as f:
    
    ilda_tf = pickle.load(f) #model
    Ingdtm_tf= pickle.load(f) #doc-token-vec
    itf_vectorizer= pickle.load(f) #vectorizer
    pyldavisItf= pickle.load(f) #pyLDAvis


#####################################################################################

import re

## USE ingredient LDA to find similar recipes:
iX_vec = itf_vectorizer.transform(recipeDF.ingred)
iX_ldaAtrans = pd.DataFrame(ilda_tf.transform(iX_vec))

from sklearn.neighbors import KDTree

simtree = KDTree(np.array(iX_ldaAtrans), metric='euclidean' )
# simtree.valid_metrics

#####################################################################################
## Earlier version -------
# def getN_easyNdiff_similarRecipes(recipeLookupIDX,Knum,SelectN):
#     dist,ind = simtree.query(np.array(iX_ldaAtrans.iloc[recipeLookupIDX]),k=Knum)
#
#     simList0 = recipeDF.ix[ind.tolist()[0]]  # [1:]]
#     simList0['KNNdist'] = dist.tolist()[0]
#     simList = simList0[1:]
#     # simList
#
#     easyOUTlist = simList[simList.prepDiffN==1].head(SelectN)
#     diffOUTlist = simList[simList.prepDiffN!=1].head(SelectN)
#
#
#     ## Might want to add a few additioanl conditions for selection -- based on a cosine distance threshold
#     ## OR use word2vec to finetune the 'in-out' categories...
#
#     # easyOUTlist = simList[simList.prepDiffN == 1].sort_values(by="KNNdist").head(ChooseN)
#     # diffOUTlist = simList[simList.prepDiffN != 1].sort_values(by="KNNdist").head(ChooseN)
#
#     return easyOUTlist, diffOUTlist

#####################################################################################

## Updated version wrt title tokens as added filter ---------

def getN_easyNdiff_similarRecipes(recipeLookupIDX, Knum, ChooseN):
    dist, ind = simtree.query(np.array(iX_ldaAtrans.iloc[recipeLookupIDX]), k=Knum)

    simList0 = recipeDF.ix[ind.tolist()[0]]  # [1:]]
    simList0['KNNdist'] = dist.tolist()[0]
    simList = simList0  # [1:]
    # simList
    ## Might want to add a few additioanl conditions for selection -- based on a cosine distance threshold
    ## OR use word2vec to finetune the 'in-out' categories...


    indx0 = set(simList.index.tolist())
    indx = indx0.difference([recipeLookupIDX])
    ref = set(simList.ix[recipeLookupIDX].title.lower().replace('&', '').replace('and', '').replace('with', '').split())
    # .replace('The ultimate makeover:','')

    keepRidx = []
    for ridx in indx:
        test = set(simList.ix[ridx].title.lower().replace('&', '').replace('and', '').replace('with', '').split())
        #.replace('The ultimate makeover:','')

        #if len(ref.intersection(test)) >= 2:
        if len(ref.intersection(test)) / len(ref) >= 0.2505:
            if "gravy" in ref and 'gravy' in test:
                keepRidx.append(ridx)
            elif "gravy" not in ref and 'gravy' not in test:
                keepRidx.append(ridx)
            elif "gravy" not in ref and 'gravy' in test:
                continue

    simList2 = simList.ix[keepRidx].sort_values(by="KNNdist")

    easyOUTlist = simList2[simList2.prepDiffN == 1].sort_values(by=["KNNdist", "title"]).head(ChooseN)
    diffOUTlist = simList2[simList2.prepDiffN != 1].sort_values(by=["KNNdist", "title"]).head(ChooseN)

    return easyOUTlist, diffOUTlist #, simList2

#####################################################################################

## Updated version wrt ingredient tokens as added filter ---------
#
# def getN_easyNdiff_similarRecipes(recipeLookupIDX, Knum, ChooseN):
#     dist, ind = simtree.query(np.array(iX_ldaAtrans.iloc[recipeLookupIDX]), k=Knum)
#
#     simList0 = recipeDF.ix[ind.tolist()[0]]  # [1:]]
#     simList0['KNNdist'] = dist.tolist()[0]
#     simList = simList0  # [1:]
#     # simList
#     ## Might want to add a few additioanl conditions for selection -- based on a cosine distance threshold
#     ## OR use word2vec to finetune the 'in-out' categories...
#
#
#     indx0 = set(simList.index.tolist())
#     indx = indx0.difference([recipeLookupIDX])
#     ref = set(
#         simList.ix[recipeLookupIDX].ingred.lower().replace('&', '').replace('and', '').replace('with', '').replace('of',
#                                                                                                                    '').split())
#
#     keepRidx = []
#     #     for ridx in indx:
#     #         test = set(simList.ix[ridx].ingred.lower().replace('&','').replace('and','').replace('with','').replace('of','').split())
#
#     #         if len(ref.intersection(test))/len(ref)>=0.3:
#     #             keepRidx.append(ridx)
#
#     Intersectpcent = []
#     for ridx in indx:
#         test = set(simList.ix[ridx].ingred.lower().replace('&', '').replace('and', '').replace('with', '').replace('of',
#                                                                                                                    '').split())
#
#         Intersectpcent.append(len(ref.intersection(test)) / len(ref))
#
#     for idx in range(len(Intersectpcent)):
#         if Intersectpcent[idx] >= np.max(Intersectpcent)*0.8:
#             keepRidx.append(list(indx)[idx])
#
#     simList2 = simList.ix[keepRidx].sort_values(by="KNNdist")
#
#     easyOUTlist = simList2[simList2.prepDiffN == 1].sort_values(by=["KNNdist", "title"]).head(ChooseN)
#     diffOUTlist = simList2[simList2.prepDiffN != 1].sort_values(by=["KNNdist", "title"]).head(ChooseN)
#
#     return easyOUTlist, diffOUTlist

#####################################################################################

# easy, diff = getN_easyNdiff_similarRecipes(recipeLookupIDX,Knum)

#####################################################################################
