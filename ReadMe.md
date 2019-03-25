
## Analysis

#### EDA:

• Target labels have great disparity almost 1:9 so cross-validation is must for this dataset  
• We can see most jobs are 1 or 2 day old and highest rejection is in the first three days but it is due to the high volume of the job has age between 0-3. The ratio of job acceptance and rejection remains constant approximately for later days.  
• Scatter plots and PCA analysis plot were created which indicated that features points are overlapping and the problem set is almost indifferentiable.

#### NANs:

• Nans are handled by using imputation through linear regression.  
• First correlation map is generated and from that most aligned features with our target feature column are taken and those aligned features are used to predict values in our target feature column. For example features main_query_tfidf, query_jl_score and query_titile_score align with feature column title_proximity_tfidf. So these features are used to predict values for feature column title_proximity_tfidf.  
• Other methods like taking mean or MICE(iterative imputer) technique was also applied but linear regression gave the best result and it was verified using cross-validation.  
• City-match column is one hot encoded, so now a total of 11 columns are there in the dataset.

#### Using 7 features:

• Now as data was highly imbalanced, I used the undersampling technique on training data so that an equal number of data of both labels are available.  
• Tomek link method was also used, but random_under_sampler performed better. This     										helped to bump the roc curve from 51.01 to 56.76.  
• Testing data remains unchanged.  
• 24 different classifiers were checked and classifiers with best scores were further fine-tuned through hyperparameter tuning using gridsearchcv.  
• Adaboost and gradient boosting classifier performed best with a score of 57.17 and 57.45 respectively. Both classifiers caught information from different feature column(found by using feature_importance) and were affect by noise, so voting was done using voting classifier(it had AdaBoost and GBC) to get final prediction(this helped to get all information from dominating features).  
• Final AUC accuracy is 57.72 %.  
• The neural network was also applied but didn’t contribute much to increment of the roc-auc curve.

#### Using classid:

• Adding a new feature makes AUC score to be 58.6% I,e ups original score by 1%.  
• No significant contribution is found, and that is also indicated by PCA and scatter plot.

### Result:


| Features | AUC_Score|
|--|--|
|7 features  | 57.74% |
|8 features(including class id)|58.71%|

## Final Section :

#### If more time is given:

• Further analysis can be done on class_id as we can surely make clusters of 157 different class-ids and get some meaningful information from it.  
• My major point will be to use SVM on this dataset if more time is given. This is not possible in short time frame as data manipulation is required and SVM can take only 1k -5k data at a time or else it gets stuck so we need to find a workaround that.  
• We can surely find a kernel space where this dataset is separable for a decent % as compared to now.
