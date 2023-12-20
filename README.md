# CET313_Assignment_BI56LZ
CET313 Artificial Intelligence assignment - Customer churn prediction using deep neural networks. <br><br>
A deep neural network was created using Keras to predict customer churn within a telecommunications company using the IBM Telco dataset 
(https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The model was then deployed as a web application using Flask.

# How to run the notebook
- Download Anaconda (https://www.anaconda.com/download)
- Open Anaconda Prompt and install the following dependencies:
   - conda install tensorflow
   - conda install python-graphviz
   - pip install imblearn
   - pip install keras-visualizer
   - pip install scikeras
   - pip install shap
- Open Jupyter Notebook through Anaconda Navigator or using the `jupyter notebook` command in the terminal.
- Locate the notebook file named "Customer Churn Prediction using Deep Neural Network.ipynb". 

# How to run the Flask application
- Download the repository as a ZIP file.
- Create virtual environment within the app directory.
- Open app directory with Visual Studio Code and activate virtual environment in the terminal.
- Install dependencies using `pip install -r requirements.txt` or `python -m pip install -r requirements.txt`
- Run the application by entering the `flask run` command in the terminal.
- Follow the link: http://127.0.0.1:5000 to open the application.

# Python version
The Anaconda distribution used to create the notebook used Python 3.9.18 and the Python version used to create the Flask web application was Python 3.10.7.
