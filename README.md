# Features
 Code for used in my final thesis in BSc.
 Visualize the correlation of environmental data and perila disease data.
 Also, predicting perila disease using machine learning methods(Logistic Regression, SVM, RF, LSTM)
 
 
# Requirement
 -Python 3.6.6
 -Keras 2.3.1
 -matplotlib 3.1.2
 -numpy 1.18.0
 -pandas 0.25.3
 -scikit-learn 0.22
 -tensorflow 1.8.0

 
# Usage
```bash
git clone https://github.com/lisashinoda/Final_thesis_BSc.git
cd data_new
python merge_data.py
  Merge the separate data as time sequence.
  
python shift.py
  Make the data suitable for machine learning methods.
  (Predict disease using by three days environmental data)
  
python filter.py
  Future selection(MethodA)
  
python corr.py
  Emit the environmental data which shows high correlation between other one.
  
python calculate_p.py
  Calculate p-value after calculation of correlation(Method B)
  
python data_graph.py
  Make counter plot graphs.
  
python logistic.py
  Start　Logistic regression and make graphs by logistic regression analysis.
  
python svm.py
  Start SVM.
  
python randomforest.py
  Start Random Forest.
  
python LSTM.py
  Start LSTM.
```
 
# Note

Excel files are supposed to create in  the "excel" file.
Visualize graphs(counter plot graphs) are in the "graph" file.
Logistic analysys graphs are in the "logisitic_graph" file.
 
# Author
* Lisa Shinoda
* Kyoto University / Agriculture
