#We import the libraries needed for our processing and calculations
import pandas as pd
import numpy as np
import math as mt 
from scipy.stats import norm
import matplotlib.pyplot as plt
#we import streamlit for the interface
import streamlit as st
from fonctions import *
import plotly.figure_factory as ff
import streamlit as st
import plotly.graph_objects as go
pd.set_option("styler.render.max_elements", 20000000)

st.set_page_config(page_title="Portfolio Management",layout="wide", initial_sidebar_state="auto", menu_items=None)


st.title("PORTFOLIO MANAGEMENT- PROJECT 1")
st.markdown("Implemented by: NADEZHDA DOBREVA & KOFFI KABLAN KAN MAX MICHEL & Supervised by: P.BOUSABAA Abdelkader")

monte_carlo, credit = st.tabs(["I-MONTE-CARLO ON OUR PORTFOLIO", "II-CREDIT DERIVATES"])
with monte_carlo:
	st.title("I-MONTE-CARLO ON OUR PORTFOLIO")
	st.header("1-PARAMETERS")
	nb_simulation, corr, component, maturity =st.columns(4)
	with nb_simulation:
		N = st.number_input("Number of simulations",value=10000)
	with corr:
		rho = st.number_input("Global correlation", value=0.11)
	with component:
		n = st.number_input("Number of componants", value=250)
	with maturity:
		m = st.number_input("Maturity(years)",  value=3)
	Xs1=np.random.normal(0,1,6) #we generate Xs for each financial sector 
	rhos1=[0.23,0.32,0.15,0.18,0.2,0.16] #correlation for each financial sector
	IC=0.99 #confidence interval level

	
	#Portfolio table view
	Portfolio=pd.read_excel("data/Credit_Portfolio.xlsx",sheet_name="Portfolio")

	#Viewing of one-year,three-years and five years probability of default  as a function of rating
	PD_Rating=pd.read_excel("data/Credit_Portfolio.xlsx",sheet_name="Params",skiprows=range(1,5), header=None).iloc[1:20,:4]
	PD_Rating.rename(columns={PD_Rating.columns[0]: 'Rating'}, inplace=True)


	#Merge the Portfolio table and the previous table to obtain the probability of default associated with each component of our portfolio
	FUSION= pd.merge(Portfolio, PD_Rating, on='Rating',how="left")

	#Table of Xs and rhos by sector
	Sector=["Utilities","Financials" ,"Energy","Consumer Non Cyclical","Technology","Consumer Cyclical"] 
	Sec_rho_X= pd.DataFrame({'Sector': Sector, 'Xs': Xs1,'rhos': rhos1})

	#Merge in between the first merge and the table of Xs and rhos by sector to obtain the Xs and rhos for each component of our portfolio
	FUSION1= pd.merge(FUSION, Sec_rho_X, on='Sector',how="left")


	#DATA 
	Xs=FUSION1["Xs"] #Xs for each component of our portfolio
	rhos=FUSION1["rhos"]  #Xs for each component of our portfolio
	EAD=FUSION1["Exposure"] #Exposure at default for each component of our portfolio
	LGD=FUSION1["LGD"] #Loss given default for each component of our portfolio
	PD1Y=FUSION1[1] #one-year Probability of default  for each component of our portfolio 
	PD3Y=FUSION1[2] #three-years Probability of default  for each component of our portfolio 
	PD5Y=FUSION1[3] #five-years Probability of default  for each component of our portfolio 
	B1Y=[norm.ppf(x) for x in PD1Y] #Default Barrier for 1-year Probability of default 
	B3Y=[norm.ppf(x) for x in PD3Y] #Default Barrier for 3-year Probability of default 
	B5Y=[norm.ppf(x) for x in PD5Y] #Default Barrier for 5-year Probability of default 

	#st.title("4-RESULTS WITH DEFAULT PROBABILITY OF 3 YEARS")

	#Results of each simulation. At the end ,we have the loss of portfolio for each simulation
	Results= MC(N, n, rho, Xs, rhos, B1Y, B3Y, B5Y, EAD, LGD, m, X, epsilons)

	#Risk indicators
	Expected_Loss=Results['Loss_Portfolio'].mean() #average of the loss distribution
	Standard_Error=Results['Loss_Portfolio'].std() #standard error of the loss distribution
	VAR=Results['Loss_Portfolio'].quantile(IC)   #Value-at-risk (quantile of order 0.99 of the loss distribution )
	Expected_ShortFall=Results['Loss_Portfolio'][Results['Loss_Portfolio'] > VAR].mean() #Expected-Shortfall (average of the losses above the var)

	statistics=pd.DataFrame({'Indicator': ['Expected_Loss','Standard_Error','VAR(IC)', 'Expected_ShortFall'],'Value': [Expected_Loss,Standard_Error,VAR, Expected_ShortFall]})

	a1, a2 = st.columns(2)
	a1.subheader('3-TABLE PORTFOLIO, DEFAULT PROBABILITY, Xs & Rhos', divider='rainbow')
	a1.table(style_dataframe(FUSION1.head(10)))
	a2.subheader('4-RESULTS WITH DEFAULT PROBABILITY OF 3 YEARS', divider='rainbow')
	a2.table(style_dataframe(Results.head(10)))


	st.subheader('INDICATORS', divider='rainbow')
	st.table(style_dataframe(statistics))
	#Result1.write(Results)
	#Result2.write(statistics)

	#Calculations to plot the graph
	#Create a matrix with N rows and 5 columns
	columns=['Rolling mean','sigma(N)','Bound-','Bound+','spread']
	val= pd.DataFrame(np.empty((N,5)), columns=columns)

	#Matrix filling
	val['Rolling mean']=[Results['Loss_Portfolio'][0:i+1].mean() for i in range(0,N)] # rolling mean of the loss distribution
	val.loc[0, ['Bound-', 'Bound+', 'spread','sigma(N)']] = 0  

	for i in range(1,N):
	    val.loc[i, 'sigma(N)'] =Results['Loss_Portfolio'][:i+1].std() #rolling standard error of the rollig mean of the loss distribution
	    val.loc[i, 'Bound-'] = val.loc[i, 'Rolling mean'] - norm.ppf(IC) * (val.loc[i, 'sigma(N)'] / mt.sqrt(i+1)) #Negative confidence interval bound
	    val.loc[i, 'Bound+'] = val.loc[i, 'Rolling mean'] + norm.ppf(IC) * (val.loc[i, 'sigma(N)'] / mt.sqrt(i+1)) #Positive confidence interval bound
	    val.loc[i, 'spread'] = val.loc[i, 'Bound+'] - val.loc[i, 'Bound-'] #spread between the confidence interval bounds
	        
	# Data to Plot the Convergence Graph

	x = list(range(N))  # the number of simulations for the x-axis

	y1 = val['Rolling mean'] #values for y-axis
	y2 = val['Bound-'] #values for y-axis 
	y3 = val['Bound+'] #values for y-axis 

	fig = go.Figure()
	# Ajouter les traces au graphique
	fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Rolling mean'))
	fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Bound-'))
	fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='Bound+'))
	# Mise en page du graphique
	fig.update_layout(
		title='Graph of convergence of the different losses of the portfolio as a function of the number of simulations',
		xaxis=dict(title='Number of simulations'),
		yaxis=dict(title='Values'),
	)
	fig.update_layout(paper_bgcolor="rgb( 238, 238, 238)",margin = {'l': 0, 'r': 50, 't': 50, 'b': 0})
	#st.plotly_chart(fig, theme="streamlit", use_container_width=True)

	b1, b2 = st.columns(2)
	b1.subheader('5-LOSS PORTFOLIO CONVERGENCE CHART', divider='rainbow')
	b1.table(style_dataframe(val.head(10)))
	b2.subheader('GRAPH OF CONVERGENCE', divider='rainbow')
	b2.plotly_chart(fig, theme="streamlit", use_container_width=True)




#seconde page
with credit:
	st.title("II-CREDIT DERIVATES")
	st.header("1-PARAMETERS")
	strike_div, size_div, m1_div =st.columns(3)
	with strike_div:cdcd
		strike = st.number_input("Strike",  value=7)
	with size_div:
		size = st.number_input("Size",  value=2)
	with m1_div:
		m_credit = st.number_input("maturity(years)",  value=5)

	#Results of each simulation. At the end ,we have the loss of portfolio for each simulation
	Results_credit= MC(N, n, rho, Xs, rhos, B1Y, B3Y, B5Y, EAD, LGD, m=m_credit, X, epsilons)
	#st.write(Results_credit)

	
	# Create a matrix with N rows and 4 columns
	columns=['loss_portfolio(%)','/Strike(%)','/Size(%)','/Tranche(%)']
	Tranche= pd.DataFrame(np.empty((N,4)), columns=columns)

	# Matrix filling
	Tranche['loss_portfolio(%)']=[(L/sum(EAD))*100 for L in Results_credit['Loss_Portfolio']] #Portfolio loss for each simulation as a percentage of the total credit granted
	Tranche['/Strike(%)']=[sorted([P, strike])[-1]-strike for P in Tranche['loss_portfolio(%)']] #This column takes 0 if the first column is below the strike or the difference between the first column and the strike if it is above the strike
	Tranche['/Size(%)']=[min(S,size) for S in Tranche['/Strike(%)']] #This column takes the minimum between the second column and the size
	Tranche['/Tranche(%)']=[(T/size)*100 for T in Tranche['/Size(%)']] #we calculate the price of the Collateralized Debt Obligation (CDO) by dividing the third column by the size
	#st.write(Tranche)


	c1, c2 = st.columns(2)
	c1.subheader('2-LOSS PORTFOLIO WITH DEFAULT PROBABILITY OF 5 YEARS', divider='rainbow')
	c1.table(style_dataframe(Results_credit.head(10)))
	c2.subheader('3-CDO SLICES \n.', divider='rainbow')
	c2.table(style_dataframe(Tranche.head(10)))


	st.subheader('4-STATISTICS', divider='rainbow')
	#Risk indicators
	Expected_Price=Tranche['/Tranche(%)'].mean()#average of  the price of the Collateralized Debt Obligation (CDO)
	Standard_Error_credit=Tranche['/Tranche(%)'].std() #standard error of the price of the Collateralized Debt Obligation (CDO)
	VAR_credit =Tranche['/Tranche(%)'].quantile(IC)  #Value-at-risk (quantile of order 0.99 of the price of the CDO  )

	statistics_credit=pd.DataFrame({'Indicator': ['Expected_Price(%)','Standard_Error(%)','VAR(IC)(%)'],'Value': [Expected_Price,Standard_Error_credit, VAR] })
	#st.write(statistics_credit)
	st.table(style_dataframe(statistics_credit))

	#st.header('5-PRICE CONVERGENCE CHART')
	#Calculations to plot the graph
	#Create a matrix with N rows and 5 columns
	columns=['Rolling mean','sigma(N)','Bound-','Bound+','spread']
	val_credit= pd.DataFrame(np.empty((N,5)), columns=columns)

	#Matrix filling
	val_credit['Rolling mean']=[Tranche['/Tranche(%)'][0:i+1].mean() for i in range(0,N)] # rolling mean of the price of the CDO
	val_credit.loc[0, ['Bound-', 'Bound+', 'spread', 'sigma(N)']] = 0  
	for i in range(1,N): 
	    val_credit.loc[i, 'sigma(N)'] = Tranche['/Tranche(%)'][:i+1].std() #rolling standard error of the rollig mean of the price of the CDO
	    val_credit.loc[i, 'Bound-'] = val_credit.loc[i, 'Rolling mean'] - norm.ppf(IC) * (val_credit.loc[i, 'sigma(N)'] / mt.sqrt(i+1)) #Negative confidence interval bound
	    val_credit.loc[i, 'Bound+'] = val_credit.loc[i, 'Rolling mean'] + norm.ppf(IC) * (val_credit.loc[i, 'sigma(N)'] / mt.sqrt(i+1)) #Positive confidence interval bound
	    val_credit.loc[i, 'spread'] = val_credit.loc[i, 'Bound+'] - val_credit.loc[i, 'Bound-'] #spread between the confidence interval bounds
	        
	# Data to Plot the Convergence Graph
	x_credit =range(N) # the number of simulations for the x-axis

	y1_credit = val_credit['Rolling mean'] #values for y-axis
	y2_credit = val_credit['Bound-'] #values for y-axis 
	y3_credit = val_credit['Bound+'] #values for y-axis 

	# Données
	x_credit = list(range(N))  # Convertir range en liste
	y1_credit = val_credit['Rolling mean']
	y2_credit = val_credit['Bound-']
	y3_credit = val_credit['Bound+']

	# Créer la figure
	fig_credit = go.Figure()
	# Ajouter les traces au graphique
	fig_credit.add_trace(go.Scatter(x=x_credit, y=y1_credit, mode='lines', name='Rolling mean'))
	fig_credit.add_trace(go.Scatter(x=x_credit, y=y2_credit, mode='lines', name='Bound-'))
	fig_credit.add_trace(go.Scatter(x=x_credit, y=y3_credit, mode='lines', name='Bound+'))
	# fig_credit en page du graphique
	fig_credit.update_layout(
		title='Graph of convergence of the different prices of the CDO as a function of the number of simulations',
		xaxis=dict(title='Number of simulations'),
		yaxis=dict(title='Values'),
	)
	fig_credit.update_layout(paper_bgcolor="rgb( 238, 238, 238)",margin = {'l': 0, 'r': 50, 't': 50, 'b': 0})

	d1, d2 = st.columns(2)
	d1.subheader('5-PRICE CONVERGENCE CHART', divider='rainbow')
	d1.table(style_dataframe(val_credit.head(10)))
	d2.subheader('GRAPH OF CONVERGENCE', divider='rainbow')
	d2.plotly_chart(fig_credit, theme="streamlit", use_container_width=True)


