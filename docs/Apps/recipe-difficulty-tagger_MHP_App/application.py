# List = [1,2,3]

import pandas as pd
import numpy as np
import re
import _json

# import loadTokenStemmerFunc
## Retrieve NLP / LDA Models
# load nltk's SnowballStemmer as variabled 'stemmer'
import nltk
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

import getSimilar
from getSimilar import getN_easyNdiff_similarRecipes

recipeDF = getSimilar.recipeDF
allRecipes = recipeDF[['title']].reset_index().to_json(orient='records')


# ### LOAD CSV File
# staticpath = '/Users/hrm/Documents/Dropbox/DSrelated/Metis/recipes/recipe-difficulty-tagger_App/static/'
# recipeListpath = staticpath + "bbcgdfood_recipelist4model_recipe-diff-tag_app.csv"
#
# recipeDF = pd.read_csv(recipeListpath)
# TitleList = recipeDF.title
# TitleList[0]
# id=9199
# # output = recipeDF.ix[id,['title','prepDifficulty','recipeLink','imgURL']]

# real idx
# indexOriginal
# title
# method
# ingred
# Ningred
# prepTime
# cookTime
# prepDifficulty
# recipeLink
# imgURL
# prepDiffN

# output = pd.read_pickle('recipeOUTout.pkl')
# output


### FLASK ###

from flask import Flask, request, render_template

app = Flask(__name__)


# @app.route("/")
# def hello():
#     return render_template('test_dropdown.html')


@app.route("/", methods=['GET','POST']) #HOMEPAGE
def displayRecipe():
    # print TitleList
    #return render_template('test_dropdown.html', L=List)
    # return render_template('test_dropdown.html', recipeDF=recipeDF, allRecipes = allRecipes)
    return render_template('test_dropdownNbootstrap.html', recipeDF=recipeDF, allRecipes = allRecipes)


# @app.route("/home/", methods=['GET','POST']) #HOMEPAGE
# def displayHomepage():
#     return render_template('test_dropdownNbootstrap2.html', recipeDF=recipeDF, allRecipes=allRecipes)

# @app.route("/findsimilar/<int:recipeID>", methods=['GET'])
@app.route("/findsimilar/<int:recipeID>", methods=['GET'])
def findsimilar(recipeID):
    #return render_template('test_dropdown.html', L=List)
    #return render_template('test_similar_recipes.html', recipeDF=output)

    #def getN_easyNdiff_similarRecipes(recipeLookupIDX, Knum, SelectN):
    easyDF, diffDF = getN_easyNdiff_similarRecipes(recipeID, 300, 3)

    #selectID = recipeID

    # return render_template('test_similar_recipes.html',
    #                        easyDF=easyDF,
    #                        diffDF=diffDF,
    #                        selectrecipeDF=recipeDF.ix[recipeID],
    #                        selectID=recipeID)

    return render_template('test_similarNbootstrap.html',
                           easyDF=easyDF,
                           diffDF=diffDF,
                           selectrecipeDF=recipeDF.ix[recipeID],
                           selectID=recipeID)

    # return "your recipe is {}".format(recipeID)



if __name__ == "__main__":
    app.run(debug=True)
