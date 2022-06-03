# -*- coding: utf-8 -*-
# +
#when we import hydralit, we automatically get all of Streamlit

import hydralit as hy
from hydralit import HydraApp
import altair as alt
import math
import os
import pandas as pd
import numpy as np  # np mean, np random
import streamlit as st  # 🎈 data web app development
#import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import apps
from hydralit import HydraApp

def load_data(nrows):
    working_directory = os.getcwd()
    filename = '\OneDrive\\Documentos\\MASTER_BIG_DATA\\Vodafone_Elena_Abril\\loan.csv'
    data_df = pd.read_csv(working_directory + filename,
                          delimiter=";")
    return data_df

raw_df = load_data(100)

app = hy.HydraApp(title='Simple Multi-Page App')

@app.addapp()
def my_home():
 hy.info('DataSet')
 num_housing = raw_df['Housing'].nunique()
 num_loan = raw_df['Loan Duration'].nunique()
 min_value = math.floor(raw_df.Age.min())
 max_value = math.ceil(raw_df.Age.max())
        # Main variables
 column1, column2, _ = st.columns([1, 1, 2])
 column1.metric("Housing:", num_housing, +10)
 column2.metric("Teams:", num_loan, '-1%')
    # Sidebar for the filters

 with st.expander("Expandir para ver datos"):
     st.markdown("## DataSet of Loan")
     st.dataframe(raw_df.head(100))

@app.addapp()
def app2():
 hy.info('Hello from app 2')


#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()
