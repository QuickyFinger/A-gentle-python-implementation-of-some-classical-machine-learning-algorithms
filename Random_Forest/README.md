## Python Implementation of Random Forest
Random forest using decision tree as a base estimator, and aggregate many trees, so called forest. It's easy to understand random forest as long as you have understood decision tree. There are two most important steps to form a tree. Firstly, using bootstrap sampling method to extract data from training data. Secondly, randomly select a subset features from feature set (m << M). Next, you can train a decision tree using extracted data and features. Repeat this step to a pre-defined iteration and you will finally get many trees. For classification, you can using voting strategy. For regression, you can simply use mean value of resulting trees. In summary, the pensodu code is described as follows:                
    
    1. Sampling data using bootstrap sampling method.               
    2. Down-sampling features from feature set (m << M).                
    3. Building decision tree using data accquied from step 1 and step 2.               
    4. Return to step 1 while the number of trees is smaller than a pre-defined value.
