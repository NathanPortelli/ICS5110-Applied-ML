# ICS5110-Applied-ML
### Project Description
This project was created in fulfilment of the Masters in AI study unit 'ICS5110 - Applied Machine Learning'. It explores the application of machine learning algorithms to predict the population of Maltese regions based on demographic factors. A full report about the project can be found in `ICS5110-Report.pdf`

```
ICS5110-Applied-ML
├── NSO_Population_Sex_dataset
│   ├── NSO_DF_TOT_POP_BY_REG_DIST_LOC_1.5.csv
│   ├── NSO_POPULATION_DATA_CLEANED.csv
│   └── NSO_POPULATION_DATA_PREFEATURE.csv
├── DecisionTreeRegressor.py
├── LinearRegression.py
├── RandomForestRegressor.py
├── Utils.py
├── app.py
├── index.html
├── main.ipynb
└── requirements.txt
└── README-gradio.md
```

### File Descriptions
- *NSO_Population_Sex_dataset*: Contains the original dataset from the National Statistics Office (NSO) containing demographic data based on gender and LAUs. Two cleaned versions of the dataset are also available, used by the machine learning algorithms and the Gradio web app.
- *DecisionTreeRegressor.py*: Custom Decision Tree implementation.
- *LineaRegression.py*: Custom Linear Regression implementation.
- *RandomForestRegressor.py*: Custom Random Forest implementation.
- *Utils.py*: Utility functions used in main.ipynb, particularly the RMSE metric.
- *app.py*: Gradio web app implementation.
- *index.html*: Contains the iframe embedding the Gradio web app running on HuggingFace onto GitHub Pages.
- *main.ipynb*: Runs the models implemented, divided into data preprocessing, model training, and model evaluation.
- *README-gradio.md*: `README.md` file required by HuggingFace containing information about the implementation, including name, SDK version, files to run etc. For running on HuggingFace, change the name of this file to `README.md`
- *requirements.txt*: Lists the Python libraries that are required to run the Gradio web app. The libraries include `gradio`, `pandas`, `numpy`, and `sklearn`.


### Project Details
- *Data Preprocessing*: The original dataset from Malta's National Statistics Office (NSO) is cleaned and preprocessed to prepare it for machine learning algorithms.
- *Machine Learning Algorithms*: Three machine learning algorithms are implemented: decision tree regression, random forest regression, and linear regression.
- *Model Evaluation*: The performance of each algorithm is evaluated using various metrics.
- *Web Application*: A user-friendly implementation of the model evaluation is created through Gradio and hosted through HuggingFace.

### Gradio Web App
A Gradio web app was created to provide a user-friendly interface for predicting population based on selected demographic factors.

### Project Resources
*Report*: Can be found above as **ICS5110-Report.pdf**
*HuggingFace*: [huggingface.co/spaces/ICS5110/Gradio-Web-Tool](https://huggingface.co/spaces/ICS5110/Gradio-Web-Tool)
*GitHub*: [nathanportelli.github.io/ICS5110-Applied-ML](https://nathanportelli.github.io/ICS5110-Applied-ML)

### Run the Gradio Web App
Install the required libraries using pip:
```
pip install -r requirements.txt
```
Run:
```
python app.py
```
Open the web app in your browser through to the URL displayed in the IDE terminal.