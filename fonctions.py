import pandas as pd
import numpy as np
import math as mt 
from scipy.stats import norm
import matplotlib.pyplot as plt
import streamlit as st

# We run a simulation with this steps

# Step 1: Generate X, the same for each component of our portfolio (in the function MC()) 

# Step 2: Generate n epsilons for each component of our portfolio  (in the function MC())

# Step 3: Generate Zi for each component of our portfolio
def generate_Z(n, rho, rhos, Xs, X, epsilons):
    Z_list = [mt.sqrt(rho) * X + mt.sqrt(rhos[i] - rho) * Xs[i] + mt.sqrt(1 - rhos[i]) * epsilons[i] for i in range(n)]
    return Z_list

# Step 4: Generate a binary variable indicating default based on maturity
def calculate_default(n, m, Z_list, B1Y, B3Y, B5Y):
    defaults = []
    for i in range(n):
        if m == 1:
            defaults.append(1 if Z_list[i] < B1Y[i] else 0)
        elif m == 3:
            defaults.append(1 if Z_list[i] < B3Y[i] else 0)
        else:
            defaults.append(1 if Z_list[i] < B5Y[i] else 0)
    return defaults

# Step 5: Generate the loss for each component of our portfolio
def calculate_loss(n, EAD, LGD, defaults):
    return [defaults[i] * EAD[i] * LGD[i] for i in range(n)]

# Step 6: Calculate the total loss of our portfolio
def calculate_total_loss(losses):
    return sum(losses)


# We run the N simulations
def MC(N, n, rho, Xs, rhos, B1Y, B3Y, B5Y, EAD, LGD, m):
    #Create a matrix with N rows and 4n+2  columns
    columns = ['X'] + [f'epsilon{i}' for i in range(1, n + 1)] + [f'Z{i}' for i in range(1, n + 1)] + \
              [f'Default{i}' for i in range(1, n + 1)] + [f'Loss{i}' for i in range(1, n + 1)] + ['Loss_Portfolio']

    matrice = pd.DataFrame(np.empty((N, 4 * n + 2)), columns=columns)
    
    #Matrix fillingcd c
    for j in range(N):
        X=np.random.normal(0, 1) #Generate X, the same for each component of our portfolio 
        epsilons=np.random.normal(0, 1,n)  #Generate n epsilons for each component of our portfolio
        Z_list =generate_Z(n, rho, rhos,Xs,X,epsilons)
        defaults= calculate_default(n, m, Z_list, B1Y, B3Y, B5Y)
        losses=calculate_loss(n, EAD, LGD, defaults)
        total_loss = calculate_total_loss(losses)

        matrice.loc[j] = [X, *epsilons, *Z_list, *defaults, *losses, total_loss]

    return matrice










def style_dataframe(df):
    # Create a style DataFrame
    styled_df = df.style
    #styled_df.set_table_attributes("style='display:inline").set_caption('Top 10 Fields of Research by Aggregated Funding Amount')

    # Color headers, indexes and body lines in midnight blue
    styled_df.set_table_styles([
        {'selector': 'td:hover','props': [('background-color', '#7dbef4')]},
        {'selector': '.index_name','props': 'font-style: italic; color: darkgrey; font-weight:normal;'},
        {'selector': 'th:not(.index_name)','props': 'background-color: #3a416c; color: white;'},
    ], overwrite=False)

    # Color the body lines alternately in gray and white
    styled_df.set_properties(**{'background-color': 'white', 'color': 'black'})
    styled_df.set_properties(subset=pd.IndexSlice[::2, :], **{'background-color': '#ebecf0'})
    styled_df.set_properties(subset=pd.IndexSlice[1::2, :], **{'background-color': '#f9fafd'})
    

    return styled_df