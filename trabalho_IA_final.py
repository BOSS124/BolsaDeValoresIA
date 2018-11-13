#!/usr/bin/env python
# coding: utf-8

# # Técnicas de I.A. aplicadas na Bolsa de Valores
# 
# ## Grupo 7
# **Igor Trevelin Xavier da Silva - 10135354 <br>**
# **Mauricio Caetano da Silva - 9040996 <br>**
# **Vitor Trevelin Xavier da Silva - 9791285 <br>**

# ## Bibliotecas Utilizadas

# In[274]:


import pandas as pd
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import linear_model
import numpy as np


# ## Funções

# In[275]:


def load_data(st_name):
    st_name += '.csv'
    data = pd.read_csv(st_name, sep='\t')
    return data


# ## Leitura da base - Pré processamento

# In[276]:


st_name = 'PETR4'
name = 'Petrobras'
dataset = load_data(st_name)
    
# Remove os caracteres especiais das colunas da tabela
dataset.columns = dataset.columns.str.replace('<','')
dataset.columns = dataset.columns.str.replace('>','')

# Transforma o tipo da coluna DATE para ser reconhecido pelo pandas
dataset['DATE'] = pd.to_datetime(dataset['DATE'])

print(name.upper()+' - '+st_name)

# Mostra apenas as 10 primeiras linhas
print(dataset.head(n=10))


# ## Remoção de colunas não interessantes

# In[277]:


# Remove colunas não interessantes da tabela
dataset_clean = dataset.drop(['TICKVOL', 'SPREAD'], axis=1)

# Mostra apenas as 10 primeiras linhas
print(dataset_clean.head(n=10))


# ## Estatísticas descritivas - Análise da base de dados

# In[278]:


print(dataset_clean.describe())


# ## Quantidade de células inválidas na base de dados

# In[279]:


if ~dataset_clean.isna().sum().all():
    dataset_clean = dataset_clean.dropna()

print(dataset_clean.isna().sum())


# 
# ## Visão geral dos valores

# In[280]:

# ## Variação da bolsa (Fechamento - Abertura)

# In[281]:


# Variação entre o preco de abertura e fechamento
dataset_clean['VARIATION'] = dataset_clean['CLOSE'].sub(dataset_clean['OPEN'])

# Mostra apenas as 10 primeiras linhas da base
print(dataset_clean.head(n=10))


# In[282]:


x = dataset_clean['DATE']
y = dataset_clean['VARIATION']

plt.plot_date(x,y, color='b',fmt="r-")
plt.xticks(rotation=30)
plt.show()


# In[283]:


# Faz uma cópia da base de dados
sample = dataset_clean.copy()
# Seleciona os 100 primeiros valores de abertura
x = sample.OPEN[:100]
# Seleciona os 100 primeiros valores de fechamento
y = sample.CLOSE[:100]
# Plota o grafico resultante
plt.scatter(x, y, color='b')
plt.xlabel('Preco de Abertura')
plt.ylabel('Preco de Fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.autoscale('False')
plt.show()


# In[284]:


# Seleciona os 100 primeiros valores de maxima
x = sample.HIGH[:100]

plt.scatter(x, y, color='b')
plt.xlabel('Preco de Maxima')
plt.ylabel('Preco de Fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.autoscale('False')
plt.show()


# In[285]:


# Seleciona os 100 primeiros valores de minima
x = sample.LOW[:100]
# Plota o grafico resultante
plt.scatter(x, y, color='b')
plt.xlabel('Preco de Mínima')
plt.ylabel('Preco de Fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.autoscale('False')
plt.show()


# In[286]:


# Seleciona os 100 primeiros valores de volume
x = sample.VOL[:100]
# Plota o grafico resultante
plt.scatter(x, y, color='b')
plt.xlabel('Volume de acoes')
plt.ylabel('Preco de Fechamento')
plt.axis([min(x),max(x),min(y),max(y)])
plt.autoscale('False')
plt.show()


# In[287]:


# Separa a classe de interesse em um novo dataframe
target = dataset_clean.CLOSE

# Variaveis independentes sao incorporados ao dataframe "independent_var"
features = ['DATE','OPEN','HIGH','LOW','VOL']
independent_var = dataset_clean[features]


# In[288]:


# Cria as variaveis de treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(independent_var, target, test_size=0.2, shuffle=False)


# In[289]:


# Cria um modelo de regressao linear
lm = linear_model.LinearRegression()
model = lm.fit(X_train[['OPEN', 'HIGH', 'LOW', 'VOL']], Y_train)
# Realiza a predicao
predictions = lm.predict(X_test[['OPEN', 'HIGH', 'LOW', 'VOL']])
predictions = np.round(predictions, 2)
r2 = r2_score(Y_test, predictions)

resultset = {
    'DATE' : X_test['DATE'],
    'OPEN' : X_test['OPEN'].ravel(),
    'HIGH' : X_test['HIGH'].ravel(),
    'LOW'  : X_test['LOW'].ravel(),
    'VOL'  : X_test['VOL'].ravel(),
    'REAL' : Y_test.ravel(),
    'PREDICTED' : predictions,
    'ERROR': Y_test.ravel() - predictions
}

print(pd.DataFrame(data=resultset).head(n=30))


# In[290]:


print('R2_Score: ', r2)