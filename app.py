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


app = hy.HydraApp(title='APP DATA SCIENCE')
#Home button will be in the middle of the nav list now
app.add_app("Data", icon="🏠", app=apps.HomeApp(title='Data'),is_home=True)
@st.cache()
def load_data(nrows):
    working_directory = os.getcwd()
    filename = '\OneDrive\\Documentos\\MASTER_BIG_DATA\\Vodafone_Elena_Abril\\loan.csv'
    data_df = pd.read_csv(working_directory + filename,
                          delimiter=";")
    return data_df

raw_df = load_data(100)

@app.addapp(is_home=True)
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

@app.addapp(title='Gráficos')
def app2():
 hy.info('Gráficos')
 st.title("Load of german3")
 st.set_option('deprecation.showPyplotGlobalUse', False)
    #histogram
 column1, column2,column3, _ = st.columns([1, 1, 1, 1])
 with st.container():
  with column1:
   df = pd.DataFrame(raw_df[:200], columns = ['Loan Amount'])
   df.hist()
   plt.show()
   st.pyplot()
   #st.sidebar.markdown("# Filters")
   #teams_selected = st.sidebar.multiselect('Select team:',
  #raw_df.Housing.unique())
    
 with st.container():
  with column2:
   df = pd.DataFrame(raw_df[:200], columns = ['Age'])
   df.hist()
   plt.show()
   st.pyplot()
    
 with st.container():
  with column3:
   df = pd.DataFrame(raw_df[:200], columns = ['Loan Duration'])
   df.hist()
   plt.show()
   st.pyplot()


@app.addapp(title='Correlación')
def app3():
 hy.info('Correlación')
 st.title("Load of german3")
 st.set_option('deprecation.showPyplotGlobalUse', False)
    #histogram
 with st.container():
  df = pd.DataFrame(raw_df[:200], columns = ['Loan Amount'])
  df.hist()
  plt.show()
  st.pyplot()
    
#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()
