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
import seaborn as sns
import sklearn
import matplotlib.cm as cm
import networkx as nx


def load_data(nrows):
    working_directory = os.getcwd()
    filename = '\OneDrive\\Documentos\\MASTER_BIG_DATA\\Vodafone_Elena_Abril\\loan.csv'
    data_df = pd.read_csv(working_directory + filename,
                          delimiter=";")
    return data_df

raw_df = load_data(1000)

def load_data2(nrows):
    working_directory = os.getcwd()
    filename = '\OneDrive\\Documentos\\MASTER_BIG_DATA\\Vodafone_Elena_Abril\\loan2.csv'
    data_df = pd.read_csv(working_directory + filename,
                          delimiter=";")
    return data_df

raw_df2 = load_data2(50)




app = hy.HydraApp(title='Data Science App')

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



@app.addapp(title='datos')
def app2(): 
 st.set_option('deprecation.showPyplotGlobalUse', False)
 column1, column2, _ = st.columns([5, 2, 0.01])
 with column1:
   st.table(raw_df.describe()) 
   with st.expander("Expandir para ver datos"):
     st.markdown("## DataSet of Loan")
     st.dataframe(raw_df.head(1000))

 with column2:
     eje_x = sns.histplot(data=raw_df, x='Age', stat='percent', color="blue")
     eje_x.set(xlabel='Age')
     eje_x.bar_label(eje_x.containers[0])
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, x='Loan Duration',  stat='percent',color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, y='Credit history',stat='percent',color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, y='Purpose',stat='percent',color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, x='Loan Amount',stat='percent',color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, y='Savings Account',stat='percent',color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, x='Age',stat='percent', color="blue")
     plt.show()
     st.pyplot()
     plt.show()
        
    
@app.addapp(title='Gráficos')
def app3():
 st.set_option('deprecation.showPyplotGlobalUse', False)
 column1, column2,_ = st.columns([5, 2, 0.01])

 with column1:
     eje_x = sns.histplot(data=raw_df, x='Age', stat='percent', color="blue")
     eje_x.set(xlabel='Age')
     eje_x.bar_label(eje_x.containers[0])  
    
 with column2:
     eje_x = sns.histplot(data=raw_df, x='Age', stat='percent', color="blue")
     eje_x.set(xlabel='Age')
     eje_x.bar_label(eje_x.containers[0])
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, x='Loan Duration',  stat='percent',color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, y='Credit history',stat='percent',color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, y='Purpose',stat='percent',color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, x='Loan Amount',stat='percent',color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, y='Savings Account',stat='percent',color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, y='Length of Current Employment',stat='percent', color="blue")
     st.pyplot(plt.show())
     eje_x = sns.histplot(data=raw_df, x='Installment Rate as % of Income',stat='percent', color="blue")
     st.pyplot(plt.show())
     eje_x = sns.histplot(data=raw_df, y='Guarantors',stat='percent', color="blue")
     st.pyplot(plt.show())
     eje_x = sns.histplot(data=raw_df, x='Length of Current Property Residence',stat='percent', color="blue")
     st.pyplot(plt.show())
     eje_x = sns.histplot(data=raw_df, y='Housing',stat='percent', color="blue")
     st.pyplot(plt.show())
     eje_x = sns.histplot(data=raw_df, x='Number of existing loans',stat='percent', color="blue")
     st.pyplot(plt.show())
     eje_x = sns.histplot(data=raw_df, y='Job',stat='percent', color="blue") 
     st.pyplot(plt.show())
     eje_x = sns.histplot(data=raw_df, x='Number of dependants',stat='percent', color="blue")
     st.pyplot(plt.show())
     eje_x = sns.histplot(data=raw_df, y='foreign worker',stat='percent', color="blue")
     st.pyplot(plt.show())
     eje_x = sns.histplot(data=raw_df, y='Sex',stat='percent', color="blue")
     st.pyplot(plt.show())
     eje_x = sns.histplot(data=raw_df, y='status',stat='percent', color="blue")
     st.pyplot(plt.show())

    
@app.addapp(title='Cluster')
def app4():
 st.set_option('deprecation.showPyplotGlobalUse', False)
 column1, column2,_ = st.columns([5, 2, 0.01])

 with column1:
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(raw_df,'ID','Target', edge_attr=True, create_using=Graphtype)
    partition = community_louvain.best_partition(G)
    # positions for all nodes
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    
    # nodes
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=1,cmap=cmap, node_color=list(partition.values()))
    
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    

    # labels
    #nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    st.pyplot()   
    
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(raw_df,'ID','Target', edge_attr=True, create_using=Graphtype)
    pos = nx.spring_layout(G)
    shapes = 'so^>v<dph8'
            # nodes
    nx.draw_networkx_nodes(G, pos, node_size=1)
    
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # labels
    #nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    st.pyplot()   
   
       
 with column2:
    with st.container():
     Variable = st.selectbox('Select variable:',("count","frequency","probability","percent", "density"))
     eje_x = sns.histplot(data=raw_df, x='Age', stat=Variable, color="blue")
     eje_x.set(xlabel='Age')
     eje_x.bar_label(eje_x.containers[0])
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, x='Loan Duration',  stat=Variable,color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, y='Credit history',stat=Variable ,color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, y='Purpose',stat=Variable,color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, x='Loan Amount',stat=Variable,color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, y='Savings Account',stat=Variable,color="blue")
     plt.show()
     st.pyplot()
     eje_x = sns.histplot(data=raw_df, x='Age',stat=Variable, color="blue")
     plt.show()
     st.pyplot()
     plt.show()
    
    
@app.addapp(title='Correlacion')
def app5():

 st.set_option('deprecation.showPyplotGlobalUse', False)

 corr = raw_df.corr()
 sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True)
 plt.show()
 st.pyplot()

 column1, column2,_ = st.columns([2, 2, 0.01])
 Variable_selected1 = st.selectbox('Select variable:',raw_df.columns)  
 Variable_selected2 = st.selectbox('Select variable:',raw_df2.columns)
 with column1:
    sns.regplot(x=raw_df[Variable_selected1], y=raw_df[Variable_selected2])
    plt.show()
    st.pyplot()  
    
 with column2:
    sns.regplot(x=raw_df[Variable_selected1], y=raw_df[Variable_selected2])
    plt.show()
    st.pyplot()

  # eje_x = sns.histplot(data=raw_df, y=Variable_selected, color="red")
  # plt.show()
   #st.pyplot()
  # eje_x = sns.histplot(data=raw_df, y=Variable_selected, color="red")
  # plt.show()
  # st.pyplot()
  # eje_x = sns.histplot(data=raw_df, y=Variable_selected, color="red")
  # plt.show()
  # st.pyplot()

#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.




@app.addapp(title='Test')
def app6():
    
    
    def create_from_pd(self, pd_graph, nx_graph=None, directional=False):

        nodes_df = pd_graph.get_nodes()
        edges_df = pd_graph.get_edges()

        # Create graph from edgelist dataframes
        if nx_graph is None:
            if directional:
                nx_graph = nx.DiGraph()
            else:
                nx_graph = nx.Graph()

        for key in edges_df:
            new_graph = nx.from_pandas_edgelist(
                edges_df[key], source="Source", target="Target", edge_attr=True)

            nx_graph = nx.compose(nx_graph, new_graph)

        # Add node attributes
        for key in nodes_df:
            df = nodes_df[key]

            for index, row in df.iterrows():
                _id = row["Id"]
                node = nx_graph.node[_id]

                for attr in row.keys():
                    node[attr] = row[attr]

        return nx_graph 


@app.addapp(title='categorizando')
def app6():

  def load_data(nrows):
      working_directory = os.getcwd()
      filename = '\OneDrive\\Documentos\\MASTER_BIG_DATA\\Vodafone_Elena_Abril\\loan.csv'
      data_df = pd.read_csv(working_directory + filename,
                          delimiter=";")
      return data_df

  df = load_data(1000)
    
  st.table(df.describe())
  st.set_option('deprecation.showPyplotGlobalUse', False)

  # Creamos las variables binarias
  dummies_Credithistory = pd.get_dummies(df['Credit history'], drop_first = True)
  dummies_Purpose = pd.get_dummies(df['Purpose'], drop_first = True)
  dummies_Savings = pd.get_dummies(df['Savings Account'], drop_first = True)
  dummies_Employment = pd.get_dummies(df['Length of Current Employment'], drop_first = True)
  dummies_Guarantors = pd.get_dummies(df['Guarantors'], drop_first = True)
  dummies_Housing = pd.get_dummies(df['Housing'], drop_first = True)
  dummies_foreign = pd.get_dummies(df['foreign worker'], drop_first = True)
  dummies_Sex = pd.get_dummies(df['Sex'], drop_first = True)
  dummies_status = pd.get_dummies(df['status'], drop_first = True)

  # Añadimos las variables binarias al DataFrame
  df = pd.concat([df, dummies_Credithistory], axis = 1)
  df = pd.concat([df, dummies_Purpose], axis = 1)
  df = pd.concat([df, dummies_Savings], axis = 1)
  df = pd.concat([df, dummies_Employment], axis = 1)
  df = pd.concat([df, dummies_Guarantors], axis = 1)
  df = pd.concat([df, dummies_Housing], axis = 1)
  df = pd.concat([df, dummies_foreign], axis = 1)
  df = pd.concat([df, dummies_Sex], axis = 1)
  df = pd.concat([df, dummies_status], axis = 1)

# Eliminamos la vairable original race
  df = df.drop(columns=['Credit history'])
  df = df.drop(columns=['Purpose'])
  df = df.drop(columns=['Savings Account'])
  df = df.drop(columns=['Length of Current Employment'])
  df = df.drop(columns=['Guarantors'])
  df = df.drop(columns=['Housing'])
  df = df.drop(columns=['foreign worker'])
  df = df.drop(columns=['Sex'])
  df = df.drop(columns=['status'])


  st.table(df.describe()) 
  st.set_option('deprecation.showPyplotGlobalUse', False)

  Graphtype = nx.Graph()
  G = nx.from_pandas_edgelist(df,'ID','Loan Amount', edge_attr=True, create_using=Graphtype)
  pos = nx.spring_layout(G)
  shapes = 'so^>v<dph8'
            # nodes
  nx.draw_networkx_nodes(G, pos, node_size=1)
    
    # edges
  nx.draw_networkx_edges(G, pos, alpha=0.5)

    # labels
    #nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
  plt.show()
  st.pyplot()   

  print(nx.info(G))
  list(G.edges(data=True))[0:1]
  list(G.nodes(data=True))[0:5]

  Graphtype = nx.Graph()
  G = nx.from_pandas_edgelist(df,'Loan Duration','Loan Amount', edge_attr='Target',create_using=Graphtype)


  fig, ax = plt.subplots(figsize=(0.51,1))
  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos, ax = ax)
  nx.draw_networkx_edges(G, pos, ax=ax)
  plt.show()
  st.pyplot()   


  df = pd.DataFrame({'number':['123','234','345'],'contactnumber':['234','345','123'],'callduration':[1,2,4]})

  df

  G = nx.from_pandas_edgelist(df,'number','contactnumber', edge_attr='Sex')



  fig, ax = plt.subplots(figsize=(12,5))
  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos, ax = ax)
  nx.draw_networkx_edges(G, pos, ax=ax)

  plt.show()
  st.pyplot()


@app.addapp(title='Escalando y Sin categorizar-Análisis')
def app6():
    
    def load_data(nrows):
      working_directory = os.getcwd()
      filename = '\OneDrive\\Documentos\\MASTER_BIG_DATA\\Vodafone_Elena_Abril\\loan.csv'
      data_df = pd.read_csv(working_directory + filename,
                          delimiter=";")
      return data_df

    df = load_data(100)
    hy.info('Análisis del tipo de dato')
    st.table(df.describe())
    
    hy.info('Nulos')
    st.table(df.isnull().sum())
    
    hy.info('Visualización por variable')
    st.pyplot(sns.pairplot(df))
    
    numerical = ['Loan amount','Age','Loan Duration', 'Number of existing loans', 'Installment Rate as % of Income','Length of Current Property Residence','Number of dependants', 'Length of Current Property Residence' ]
    categorical = ['Credit history','Sex','Housing','Savings Account','Purpose','Job', 'status']
    unused = ['ID: 0']
    hy.info('Análisis varible categóricas')
    fig = plt.figure(figsize = (20,15))
    axes = 430
    
    for cat in categorical:
        
     axes += 1
     fig.add_subplot(axes)
     sns.countplot(data = df, x = cat)
     plt.xticks(rotation=30)
        
    st.pyplot(plt.show())
    
    hy.info('Cluster')
    df_cluster = pd.DataFrame()
    df_cluster['Loan Amount'] = df['Loan Amount']
    df_cluster['Age'] = df['Age']
    df_cluster['Loan Duration'] = df['Loan Duration']
    df_cluster['Job'] = df['Job']
    st.table(df_cluster.head())

    df_cluster_log = np.log(df_cluster[['Age', 'Loan Amount','Loan Duration']])
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
    sns.distplot(df_cluster_log["Age"], ax=ax1)
    sns.distplot(df_cluster_log["Loan Amount"], ax=ax2)
    sns.distplot(df_cluster_log["Loan Duration"], ax=ax3)
    st.pyplot(plt.tight_layout())

     #fit and transform
    df_cluster_log.head()
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(df_cluster_log)
    #k-means
    from sklearn.cluster import KMeans
    hy.info('K-Means')
    Sum_of_squared_distances = []
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(cluster_scaled)
        Sum_of_squared_distances.append(km.inertia_)
    plt.figure(figsize=(20,5))
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    st.pyplot(plt.show())
    
    from mpl_toolkits.mplot3d import Axes3D

    model = KMeans(n_clusters=3)
    model.fit(cluster_scaled)
    kmeans_labels = model.labels_

    fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    ax = plt.axes(projection="3d")

    ax.scatter3D(df_cluster['Age'],df_cluster['Loan Amount'],df_cluster['Loan Duration'],c=kmeans_labels, cmap='rainbow')

    xLabel = ax.set_xlabel('Age', linespacing=3.2)
    yLabel = ax.set_ylabel('Loan Amount', linespacing=3.1)
    zLabel = ax.set_zlabel('Loan Duration', linespacing=3.4)
    print("K-Means")
    st.pyplot(plt.show())
    
    #Gráfos
    hy.info('Gráfos-Loan Duration Loan Amount Age')
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df,'Loan Duration','Loan Amount', edge_attr='Age', create_using=Graphtype)
    partition = community_louvain.best_partition(G)
    # positions for all nodes
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    
    # nodes
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=1,cmap=cmap, node_color=list(partition.values()))
    
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    

    # labels
    #nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    st.pyplot()   
    

    #cluster
    hy.info('Gráfos-Loan Duration Loan Amount Purpose')
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df,'Loan Duration','Loan Amount', edge_attr='Purpose', create_using=Graphtype)
    partition = community_louvain.best_partition(G)
    # positions for all nodes
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    
    # nodes
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=1,cmap=cmap, node_color=list(partition.values()))
    
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    

    # labels
    #nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    st.pyplot()   
    
    #cluster
    hy.info('Gráfos-LAge Loan Amount Loan Duration')
    Graphtype = nx.Graph()
    G = nx.from_pandas_edgelist(df,'Age','Loan Amount', edge_attr='Loan Duration', create_using=Graphtype)
    partition = community_louvain.best_partition(G)
    # positions for all nodes
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    
    # nodes
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=1,cmap=cmap, node_color=list(partition.values()))
    
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    

    # labels
    #nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    st.pyplot()   
    
    #cluster
    hy.info('Gráfos-Loan Duration Loan Amount-True')
    G = nx.from_pandas_edgelist(df,'Loan Duration','Loan Amount', edge_attr=True, create_using=Graphtype)
    partition = community_louvain.best_partition(G)
    # positions for all nodes
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    
    # nodes
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=1,cmap=cmap, node_color=list(partition.values()))
    
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    

    # labels
    #nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    st.pyplot() 
    
    
    #cluster
    hy.info('Gráfos-ID Age True')
    G = nx.from_pandas_edgelist(df,'ID','Age', edge_attr=True, create_using=Graphtype)
    partition = community_louvain.best_partition(G)
    # positions for all nodes
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    
    # nodes
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=1,cmap=cmap, node_color=list(partition.values()))
    
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    

    # labels
    #nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    st.pyplot()  
    
     #cluster
    hy.info('Gráfos-ID Target True')
    G = nx.from_pandas_edgelist(df,'ID','Target', edge_attr=True, create_using=Graphtype)
    partition = community_louvain.best_partition(G)
    # positions for all nodes
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    
    # nodes
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=1,cmap=cmap, node_color=list(partition.values()))
    
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    

    # labels
    #nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    st.pyplot() 
    
        #cluster
    hy.info('Gráfos-Target ID True')
    G = nx.from_pandas_edgelist(df,'Target','ID', edge_attr=True, create_using=Graphtype)
    partition = community_louvain.best_partition(G)
    # positions for all nodes
    pos = nx.spring_layout(G)
    # color the nodes according to their partition
    cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
    
    # nodes
    nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=1,cmap=cmap, node_color=list(partition.values()))
    
    # edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    

    # labels
    #nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    plt.show()
    st.pyplot() 
   
           

    
    
app.run()
