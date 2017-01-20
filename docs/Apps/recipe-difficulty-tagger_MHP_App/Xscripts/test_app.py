import os
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug import secure_filename
import numpy as np
from sklearn.externals import joblib
# from annoy import AnnoyIndex
import random
import pickle

import os

import pandas as pd
import numpy as np



###-------------------
# filepath

# modelpath = '/Users/hrm/Documents/Dropbox/DSrelated/Metis/recipes/recipeApp/models/'

###-------------------

# Initialize the Flask application
app = Flask(__name__)


# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    # return render_template('recipe_index.html')
    return render_template('test_index.html')




@app.route("/recommend/", methods=["POST"])
def recommend():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get value for our example that came with the request
    data = request.json

    # prob_nmf = pickle.load(open('models/prob_nmf.pickle', 'rb'))
    # # prob_nmf = joblib.load('models/prob_nmf.pkl')
    # all_titles = pickle.load(open('models/all_titles.pkl', 'rb'))

    # f = 30
    # t = AnnoyIndex(f)  # Length of item vector that will be indexed
    # for i, row in enumerate(prob_nmf):
    #     v = row
    #     t.add_item(i, v)
    #
    # t.build(10) # 10 trees


    ###########
    title = data["example"].strip('\"')


    # w2vTITLE = utils.unpickle(modelpath+"w2vTitle_s410_minC40pcent_window7.model")
    # # w2vTITLE = utils.unpickle(modelpath + "w2vTitle_s400_minC60pcent_window7.model")
    # DF2 = pd.read_pickle(modelpath+'BBCgoodfood_TokensNLemms4word2vec.pkl')
    #
    # outlist = [[i, round(v * 1000) / 1000] for i, v in w2vTITLE.most_similar(positive=[title], topn=200)
    #            if i not in [n for m in DF2.ingredLems for n in m] and i not in ['BBC Children in Need cupcakes']
    #            and v > 0.76]
    # outlist[:5]


    searchedTitle= [title]
    # RECrecipes = outlist[:5] #['test rec 0','test rec 1','test rec 2']


    # blog_list = ["", ""]
    # count = 0
    # for i in idx:
    #     if all_titles[i][:5] == 'IDEA_':
    #         blog_list[count] = all_titles[i][5:]
    #         count += 1
    #         if count > 1:
    #             break

    # Put the result in a nice dict so we can send it as json
    # results = {"recommend_tedx": tedx_list,
    #            "recommend_blog": blog_list}

    results = {"searchedTitle": searchedTitle}

    # results = {"searchedTitle": searchedTitle,
    #            "RECrecipes": RECrecipes}

    return jsonify(results)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8808, debug=True)





# ### --------------------------------------------------------------------------
