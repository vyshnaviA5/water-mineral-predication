#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv(r"C:\Users\VivoBook\Downloads\archive (5)\water.csv")
df


# In[2]:


df.head()


# In[3]:


df.tail()


# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')


# In[5]:


df.info()


# In[6]:


df.describe().T


# In[7]:


df.dtypes.tolist()


# In[8]:


columns_list=df.columns.tolist()
print(columns_list)


# In[9]:


df.isna().sum()


# In[10]:


df.duplicated().sum()


# In[11]:


medians=df.groupby('Potability').median()
medians


# In[12]:


missing_col=df.isnull().sum().sort_values(ascending=False)
print(missing_col)


# In[13]:


import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
fig = go.Figure(data=[go.Pie(
    labels=missing_col.index,
    values=missing_col.values,
    hole=0.4,
    marker=dict(colors=['orange', 'lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']),
)])

fig.update_traces(textposition='inside', textinfo='percent+label')
fig.update_layout(title='Missing Values Distribution by Column', template='plotly_white')

fig.show()

## Imputing missing values
imputer = SimpleImputer(strategy='mean')
df['Sulfate'] = imputer.fit_transform(df[['Sulfate']])
df['ph'] = imputer.fit_transform(df[['ph']])
df['Trihalomethanes'] = imputer.fit_transform(df[['Trihalomethanes']])
df.isnull().sum()


# In[14]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
sns.histplot(df['ph'])
plt.show()


# In[15]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
sns.histplot(df['Hardness'])
plt.show()


# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
sns.histplot(df['Solids'])
plt.show()


# In[17]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
sns.histplot(df['Chloramines'])
plt.show()


# In[18]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
sns.histplot(df['Sulfate'])
plt.show()


# In[19]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
sns.histplot(df['Conductivity'])
plt.show()


# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
sns.histplot(df['Organic_carbon'])
plt.show()


# In[21]:


potability_distribution = df['Potability'].value_counts().reset_index()
potability_distribution.columns = ['Potability', 'Count']
# map potability values to labels
potability_distribution['Potability'] = potability_distribution['Potability'].map({0: 'Non-potable', 1: 'Potable'})

# create a bar plot using Plotly
fig = go.Figure(go.Bar(
    x=potability_distribution['Potability'],
    y=potability_distribution['Count'],
    marker=dict(color=['orange', '#3498DB'])
))

# Update layout
fig.update_layout(
    title='Distribution of Potability in Water Quality Dataset',
    xaxis=dict(title='Potability'),
    yaxis=dict(title='Number of Samples'),
    template='plotly_dark',
    margin=dict(l=50, r=50, t=100, b=50),
)

fig.show()


# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
sns.histplot(df['ph'])
plt.show()


# In[23]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
sns.histplot(df['Hardness'])
plt.show()


# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
plt.boxplot(df['Solids'])
plt.show()


# In[29]:


import pandas as pd
import plotly.express as px
correlation = df['ph'].corr(df['Potability'])
fig = px.scatter(df, x='Potability', y='ph', trendline='ols',
                 title=f'Correlation between pH and Potability (Correlation coefficient: {correlation:.2f})')
fig.update_layout(xaxis_title='Potability', yaxis_title='pH')
fig.show()


# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
correlation_coefficient=df['Hardness'].corr(df['Potability'])
plt.scatter(df['Hardness'],df['Potability'])
plt.title(f'Scatter Plot (Correlation: {correlation_coefficient:.2f})')
plt.xlabel('Hardness')
plt.ylabel('Potability')
plt.show()


# In[35]:


correlation = df['ph'].corr(df['Potability'])
fig=px.scatter(df,x='Potability',y='ph',trendline='ols',
               title=f'correlation between ph and potability (correlation coefficient:{correlation:.2f})')
fig.update_layout(xaxis_title='Potability',yaxis_title='pH')
fig.show()
##### correlation between hardness and potability
correlation=df['Hardness'].corr(df['Potability'])
fig=px.scatter(df,x='Potability',y='Hardness',trendline='ols',
               title=f'correlation between hardness and Potability (correlation coefficient: {correlation:.2f})')
fig.update_layout(xaxis_title='Potability',yaxis_title='Hardness')
fig.show()
# calculate correlation matrix 
correlation_matrix = df.corr()
# plot correlation matrix heatmap
fig=px.imshow(correlation_matrix,
              labels=dict(x='features', y='features', color='correlation'),
              x=correlation_matrix.index,
              y=correlation_matrix.columns,
              color_continuous_scale='Viridis',
              title='correlation matrix heatmap')
fig.show()


# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
plt.scatter(df['Sulfate'],df['Trihalomethanes'])
plt.xlable=('Sulfate')
plt.ylabel=('Trihalomethanes')
plt.show()


# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
plt.scatter(df['Sulfate'],df['Turbidity'])
plt.xlabel=('Sulfate')
plt.ylabel=('Turbidity')
plt.show()


# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
plt.scatter(df['Conductivity'],df['Organic_carbon'])
plt.xlabel=('Conduvtivity')
plt.ylabel=('Organic_carbon')
plt.show()


# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
url=r"C:\Users\VivoBook\Downloads\archive (5)\water.csv"
df=pd.read_csv(url)
plt.scatter(df['Trihalomethanes'],df['Conductivity'])
plt.xlabel=('Trihalomethanes')
plt.ylabel=('Conductivity')
plt.show()


# In[44]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest Classifier': RandomForestClassifier(),
    'SVM': SVC(), 
    'Gradient Boosting': GradientBoostingClassifier(),
    'Neural Network': MLPClassifier()
}


# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data into X and y
X = df.drop(columns=['Potability'])
y = df['Potability']

# Split data into train test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[51]:


# Drop rows with missing values
df.dropna(inplace=True)

# Split data into X and y
X = df.drop(columns=['Potability'])
y = df['Potability']

# Split data into train test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Training and evaluating models
for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'Confusion Matrix': cm
    })
    results_df = pd.DataFrame(results)


# In[54]:


print("Predictive modeling results for water potability Prediction")
print(results_df)


# In[53]:


import seaborn as sns

plt.figure(figsize=(12, 8))
plt.suptitle('Model Performance Metrics', fontsize=20, fontweight='bold')

# Accuracy Plot
plt.subplot(2, 2, 1)
sns.barplot(x='Model', y='Accuracy', data=results_df, color='orange')
plt.title('Accuracy', fontsize=16)
plt.xticks(rotation=45, ha='right')

# Precision Plot
plt.subplot(2, 2, 2)
sns.barplot(x='Model', y='Precision', data=results_df, color='orange')
plt.title('Precision', fontsize=16)
plt.xticks(rotation=45, ha='right')

# Recall Plot
plt.subplot(2, 2, 3)
sns.barplot(x='Model', y='Recall', data=results_df, color='orange')
plt.title('Recall', fontsize=16)
plt.xticks(rotation=45, ha='right')

# F1 Score Plot
plt.subplot(2, 2, 4)
sns.barplot(x='Model', y='F1 Score', data=results_df, color='orange')
plt.title('F1 Score', fontsize=16)
plt.xticks(rotation=45, ha='right')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# In[ ]:




