# Importing the requisite libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *

# Importing the dataset
git_url='https://raw.githubusercontent.com/debadrita1517/The-Sparks-Foundation-Internship-Tasks/main/Task-3-Exploratory%20Data%20Analysis-Retail/SampleSuperstore.csv'
sample=pd.read_csv(git_url)
sample.head(5)

# Verifying the missing values
sample.isnull().sum()

# Getting the information of the data
sample.info()

# Getting the description of the variables
sample.describe()

# Deleting the variable 'Postal Code'
col=['Postal Code']
sample1=sample.drop(columns=col,axis=1)

# Showing the correlation between the variables
sample1.corr()

sample1.hist(bins=50,figsize=(20,15))
plt.show();

# Showing the gain and loss of each and every subcategory in this business course
profit_plot=(ggplot(sample,aes(x='Sub-Category',y='Profit',fill='Sub-Category'))+geom_col()+coord_flip()+scale_fill_brewer(type='div',palette="Spectral")+theme_classic()+ggtitle('Pie Chart'))
display(profit_plot)

sns.set(style="whitegrid")
plt.figure(2,figsize=(20,15))
sns.barplot(x='Sub-Category',y='Profit',data=sample,palette='Spectral')
plt.suptitle('PIE CONSUMPTION PATTERNS IN THE UNITED STATES',fontsize=16)

# Plotting the Bar Graph showing the Ship Mode
ggplot(sample,aes(x='Ship Mode',fill='Category'))+geom_bar(stat='count')

# Pair Plotting the sub-category variables
figsize=(15,10)
sns.pairplot(sample1,hue='Sub-Category')

flip_xlabels=theme(axis_text_x=element_text(angle=90, hjust=1),figure_size=(10,5),
                     axis_ticks_length_major=10,axis_ticks_length_minor=5)
(ggplot(sample, aes(x='Sub-Category', fill='Sales')) + geom_bar() + facet_wrap(['Segment']) 
+ flip_xlabels +theme(axis_text_x = element_text(size=12))+ggtitle("Sales From Every Segment Of United States of Whole Data"))

flip_xlabels=theme(axis_text_x = element_text(angle=90, hjust=1),figure_size=(10,5),
                     axis_ticks_length_major=10,axis_ticks_length_minor=5)
(ggplot(sample, aes(x='Sub-Category', fill='Discount')) + geom_bar() + facet_wrap(['Segment']) 
+ flip_xlabels +theme(axis_text_x = element_text(size=12))+ggtitle("DISCOUNT ON CATEGORIES FROM EVERY SEGMENT OF THE WHOLE DATA OF THE UNITED STATES"))

flip_xlabels = theme(axis_text_x = element_text(angle=90, hjust=10),figure_size=(10,10),axis_ticks_length_major=5,
                     axis_ticks_length_minor=5)
(ggplot(sample, aes(x='Category', y='Sales')) + geom_boxplot() + theme(axis_text_x = element_text(size=10)) 
+ facet_wrap(['State']) + flip_xlabels + coord_cartesian(ylim = (0, 2000))+ggtitle("Sales From Every State Of United States Of Whole Data"))

flip_xlabels = theme(axis_text_x=element_text(angle=90, hjust=10),figure_size=(10,10),
                     axis_ticks_length_major=5,axis_ticks_length_minor=5)
(ggplot(sample, aes(x='Category', y='Profit')) + geom_boxplot() + theme(axis_text_x = element_text(size=10)) 
+ facet_wrap(['State']) + flip_xlabels+coord_cartesian(ylim = (-4000,5000))+ggtitle("PROFIT/LOSS IN EVERY STATE OF THE WHOLE DATA OF THE UNITED STATES"))

flip_xlabels=theme(axis_text_x = element_text(angle=90, hjust=10),figure_size=(10,10),
                     axis_ticks_length_major=50,axis_ticks_length_minor=50)
(ggplot(sample, aes(x='Category', fill='Sales')) + geom_bar() + theme(axis_text_x = element_text(size=10)) 
+ facet_wrap(['Region']) + flip_xlabels+ ggtitle("SALES FROM EVERY REGION OF THE WHOLE DATA OF THE UNITED STATES"))

flip_xlabels=theme(axis_text_x = element_text(angle=90, hjust=10),figure_size=(10,10),axis_ticks_length_major=5,axis_ticks_length_minor=5)
(ggplot(sample, aes(x='Region', fill='Quantity')) + geom_bar() + theme(axis_text_x = element_text(size=10)) 
+ facet_wrap(['Discount']) + flip_xlabels+ggtitle("DISCOUNT ON THE NUMBER OF QUANTITY FROM SALES OF THE UNITED STATES"))

flip_xlabels=theme(axis_text_x = element_text(angle=90, hjust=10),figure_size=(10,10),axis_ticks_length_major=5,axis_ticks_length_minor=5)
(ggplot(sample, aes(x='Category', y='Discount')) + geom_boxplot() + theme(axis_text_x = element_text(size=10)) 
+ facet_wrap(['State']) + flip_xlabels+ggtitle("DISCOUNT ON CATEGORIES ON EVERY STATE OF THE WHOLE DATA OF THE UNITED STATES"))

grouped=pd.DataFrame(sample.groupby(['Ship Mode','Segment','Category','Sub-Category','State','Region'])['Quantity','Discount','Sales','Profit'].sum().reset_index())
grouped

# sum, mean, min, max, count, median, standard deviation, variance of each states having Profit
sample.groupby("State").Profit.agg(["sum","mean","min","max","count","median","std","var"])

# Displaying sales from every region of the US after grouping
flip_xlabels = theme(axis_text_x = element_text(angle=90, hjust=10),figure_size=(10,10),axis_ticks_length_major=5,axis_ticks_length_minor=5)
(ggplot(grouped, aes(x='Category', fill='Sales')) + geom_bar() + theme(axis_text_x = element_text(size=10)) 
 + facet_wrap(['Region']) + flip_xlabels+ggtitle("SALES FROM EVERY REGION OF THE UNITED STATES AFTER GROUPING"))

# Displaying sales from every state of the US after grouping
flip_xlabels = theme(axis_text_x = element_text(angle=90, hjust=10),figure_size=(10,10),axis_ticks_length_major=5,axis_ticks_length_minor=5)
(ggplot(grouped, aes(x='Category', y='Sales')) + geom_boxplot() + theme(axis_text_x = element_text(size=10)) 
+ facet_wrap(['State']) + flip_xlabels+ggtitle("SALES FROM EVERY STATE OF THE UNITED STATES AFTER GROUPING"))

# Displaying profit/loss from every state of the US after grouping
flip_xlabels = theme(axis_text_x = element_text(angle=90, hjust=10),figure_size=(10,10),axis_ticks_length_major=5,axis_ticks_length_minor=5)
(ggplot(grouped, aes(x='Category', y='Profit')) + geom_boxplot() + theme(axis_text_x = element_text(size=10)) 
+ facet_wrap(['State']) + flip_xlabels+ggtitle("PROFIT/LOSS FROM EVERY STATE OF THE UNITED STATES AFTER GROUPING"))

# Displaying the discount given on categories from every state of the US after grouping
flip_xlabels = theme(axis_text_x = element_text(angle=90, hjust=10),figure_size=(10,10),axis_ticks_length_major=5,axis_ticks_length_minor=5)
(ggplot(grouped, aes(x='Category', y='Discount')) + geom_boxplot() + theme(axis_text_x = element_text(size=10)) 
+ facet_wrap(['State']) + flip_xlabels+ggtitle("DISCOUNT GIVEN ON CATEGORIES FROM EVERY STATE OF THE UNITED STATES AFTER GROUPING"))

x=sample.iloc[:,[9,10,11,12]].values
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0).fit(x)
  wcss.append(kmeans.inertia_)

# Plotting the 'Sales' and 'Quantity' on the sub-categories
sns.set_style("whitegrid")
sns.FacetGrid(sample,hue="Sub-Category",height=6).map(plt.scatter,'Sales','Quantity')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')
plt.legend()

# Plotting the 'Sales' and 'Profit' on the sub-categories
sns.set_style("whitegrid")
sns.FacetGrid(sample,hue="Sub-Category",height=6).map(plt.scatter,'Sales','Profit')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='yellow',label='Centroids')
plt.legend()
