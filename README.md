# Features
Code, which is used in my final thesis in BSc.<br>
Visualize the correlation of environmental data and perila disease data.<br>
Also, predicting perila disease using machine learning methods(Logistic Regression, SVM, RF, LSTM)
 
 
# Requirement
* Python 3.6.6
* Keras 2.3.1
* matplotlib 3.1.2
* numpy 1.18.0
* pandas 0.25.3
* scikit-learn 0.22
* tensorflow 1.8.0


 
# Usage
```bash
git clone https://github.com/lisashinoda/Final_thesis_BSc.git
cd data_new
```
### Merge the separate data as time sequence.
```bash
python merge_data.py
```
### Make the data suitable for machine learning methods.
### (Predict disease using three days environmental data)

```bash
python shift.py
```
 
### Future selection(MethodA)
```bash
python filter.py
```

### Emit the environmental data which shows high correlation between the other one.
```bash
python corr.py
```
  
### Calculate p-value after calculation of correlation(Method B)
```bash
python calculate_p.py
```
  
### Make counter plot graphs.
```bash
python data_graph.py
```
  
### Start　Logistic regression and make graphs by logistic regression analysis.
```bash
python logistic.py
```
  
### Start SVM.
```bash
python svm.py
```

### Start Random Forest.
```bash
python randomforest.py
```
 
### Start LSTM.
```bash
python LSTM.py
```
 
# Note
Excel files are supposed to create in  the "excel" file.<br>
Visualize graphs(counter plot graphs) are in the "graph" file.<br>
Logistic analysys graphs are in the "logisitic_graph" file.
 
# Author
* Lisa 
* Kyoto University 
