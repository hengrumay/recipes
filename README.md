# metis_project_recipes 
** (folder being updated)  

I love to cook but I am quite terrible at following recipes -- I tend to modify them! (possible recipe-dyslexia).  
 
However, I love getting inspirations from different recipes and foodie experience.   
I wanted to explore how one could come up with alternative recipe suggestions 
- a) based on similarity of ingredients within a recipe of interest,   
and whether  
- b) we could use ingredients and instructions from recipes to categorize their 'difficulty'.

Recipe data was scraped from bbcgoodfood.com.

The file(s) in this folder are associated with the 2 web apps developed in trying to answer the 2 specific interests described above, which involved Natural Language Processing (NLP). They are summarized in the presentations linked below:  
- 1) [Recipe Suggestion] (https://github.com/hengrumay/metis_project_recipes/blob/master/docs/RecipeSuggestor.pptx) -- Recipe suggestions (a minimal web app) are offered based on similarity in ingredient list modelled and trained using [word2vec](https://code.google.com/archive/p/word2vec/) (a shallow Artifial Neural Network for NLP).   
- 2) [Menu (Helper) Planner] (https://github.com/hengrumay/metis_project_recipes/blob/master/docs/H-RM_MenuHelper_v2.pptx) -- A "Recipe-Difficulty-Tagger" was developed using CRF and Topic Modelling to engineer features for classifying recipe preparation difficulty (~86% precision & recall) and created a web-app to offer recipe alternatives based on clustering by topic similarity and categorized by difficulty.  
