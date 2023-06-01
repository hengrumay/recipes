## Menu Planner Helper:   
***an app to provide recipe alternatives based on similar recipes and categorized by recipe difficulty***
###  

I love to cook but I have a tendency to modify recipes!
As someone who derives inspiration from different recipes and foodie experience, I wanted to explore how one could come up with alternative recipe suggestions 
- a) based on similarity of ingredients within a recipe of interest,   
and whether we could  
- b) use ingredients and instructions from recipes to categorize their 'difficulty'

Recipe data was scraped from [bbcgoodfood.com](www.bbcgoodfood.com)

The file(s) in the [doc](https://github.com/hengrumay/recipes/tree/master/docs) folder are associated with the 2 web apps developed in trying to answer the 2 specific interests described above, which involved Natural Language Processing (NLP). They are summarized in the presentations linked below:  
- 1) [Recipe Suggestion](https://github.com/hengrumay/metis_project_recipes/blob/master/docs/RecipeSuggestor.pptx.pdf) -- Recipe suggestions (a minimal web app) are offered based on similarity in ingredient list modelled and trained using [word2vec](https://code.google.com/archive/p/word2vec/) (a shallow Artificial Neural Network for NLP).   

- 2) [Menu (Helper) Planner](https://github.com/hengrumay/recipes/blob/master/docs/MayMerkle-Tan_MenuPlannerHelper_updated.pdf) -- A "Recipe-Difficulty-Tagger" was developed using [Conditional Random Fields (CRF)](http://homepages.inf.ed.ac.uk/csutton/publications/crftutv2.pdf), in particular [NYT's ingredient phrase CRF model](https://open.blogs.nytimes.com/2016/04/27/structured-ingredients-data-tagging/) to structure the ingredient information, and [Latent Dirichlet Allocation (LDA)](http://www.cs.princeton.edu/~blei/papers/Blei2012.pdf) Topic Modelling to engineer features for classifying recipe preparation difficulty (~86% precision & recall on hold-out) and created a prototype flask [web-app](http://bit.ly/menuplannerhelper) to offer recipe alternatives based on [(K Nearest Neighbours)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) clustering by topic similarity and categorized by difficulty.  

Check out the write up of this work in this [blog-post](https://hengrumay.github.io/MenuPlannerHelper/), which was hightlighted on [www.datascienceweekly.org](https://www.datascienceweekly.org/newsletter/data-science-weekly-newsletter-issue-215)!
