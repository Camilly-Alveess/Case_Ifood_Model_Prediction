import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import *
import csv
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config( page_title = 'Case Ifood',
                    page_icon = './images/fiap.png',
                    layout = 'wide',
                    initial_sidebar_state = 'expanded')

st.title('Conversão de Vendas')
with st.expander('Descrição do App', expanded = False):
    st.write('O aplicativo usa dados históricos do iFood para prever o comportamento dos clientes, incluindo preferências alimentares e participação em campanhas. Com uma interface intuitiva, fornece insights estratégicos para decisões personalizadas e inteligentes, simplificando a análise de dados e capacitando os usuários a otimizar seus negócios no iFood.')

tabs = st.tabs(['Predições', 'Analytics'])


with st.sidebar:
    c1, c2 = st.columns(2)
    c1.image('./images/imagem.png', width = 100)
    c2.write('')
    c2.subheader('App iFood')

    database = st.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'))

    if database == 'CSV':
        st.info('Upload do CSV')
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')


#Tela principal
if database == 'CSV':
    with tabs[0]:
        st.write("Predições")
        if file:
            #carregamento do CSV
            Xtest = pd.read_csv(file)

            #carregamento / instanciamento do modelo pkl
            tuned_lgbm = load_model('./pickle/pickle_et_pycaret')

            #predict do modelo
            ypred = predict_model(tuned_lgbm, data = Xtest, raw_score = True)

            with st.expander('Visualizar CSV carregado:', expanded = False):
                c1, _ = st.columns([2,4])
                qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:', 
                                        min_value = 5, 
                                        max_value = Xtest.shape[0], 
                                        step = 10,
                                        value = 5)
                st.dataframe(Xtest.head(qtd_linhas))

            with st.expander('Visualizar Predições:', expanded = True):
                c1, _, c2, c3 = st.columns([2,.5,1,1])
                treshold = c1.slider('Treshold (ponto de corte para considerar predição como True)',
                                    min_value = 0.0,
                                    max_value = 1.0,
                                    step = .1,
                                    value = .5)
                qtd_true = ypred.loc[ypred['prediction_score_1'] > treshold].shape[0]

                c2.metric('Clientes True', value = qtd_true)
                c3.metric('Clientes False', value = len(ypred) - qtd_true)
                
                def color_pred(val):
                    color = 'green' if val > treshold else 'red'
                    return f'background-color: {color}'

                tipo_view = st.radio('', ('Completo', 'Apenas predições'))
                if tipo_view == 'Completo':
                    df_view = ypred.copy()
                else:
                    df_view = pd.DataFrame(ypred.iloc[:,-1].copy())

                st.dataframe(df_view.style.applymap(color_pred, subset = ['prediction_score_1']))

                csv = df_view.to_csv(sep = ';', decimal = ',', index = True)
                st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
                st.download_button(label = 'Download CSV',
                                data = csv,
                                file_name = 'Predicoes.csv',
                                mime = 'text/csv')

            
        else:
            st.warning('Arquivo CSV não foi carregado')

    with tabs[1]:
        if file:
            fig = plt.figure(figsize=(10, 6))
            sns.barplot(x='Education', y='prediction_label', data=ypred, palette='coolwarm', order=['2n Cycle', 'Basic', 'Graduation', 'Master', 'PhD'])
            plt.title('Relação entre Graduação e Resultado Previsto')
            plt.xlabel('Nível de Educação')
            plt.ylabel('Resultado Previsto')
            st.pyplot(fig)
        else:
            st.warning('Arquivo CSV não foi carregado')

else:
        AcceptedCmp1 = st.text_input(label="Aceitou Campanha 1", value="")
        AcceptedCmp2 = st.text_input(label="Aceitou Campanha 2", value="")
        AcceptedCmp3 = st.text_input(label="Aceitou Campanha 3", value="")
        AcceptedCmp4 = st.text_input(label="Aceitou Campanha 4", value="")
        AcceptedCmp5 = st.text_input(label="Aceitou Campanha 5", value="")
        Age= st.text_input(label="Idade", value="")
        Complain = st.text_input(label="Reclamação", value="")
        Education = st.text_input(label="Educação", value="")
        Income = st.text_input(label="Renda", value="")
        Kidhome = st.text_input(label="Qtd Crianças em casa", value="")
        Marital_Status = st.text_input(label="Estado Civil", value="")
        MntFishProducts = st.text_input(label="Quantidade de Produtos sendo Peixes", value="")
        MntFruits = st.text_input(label="Quantidade de Produto sendo Frutas", value="")
        MntGoldProds = st.text_input(label="Quantidade de Produto sendo Alto padrão", value="")
        MntMeatProducts = st.text_input(label="Quantidade de Produto sendo Carnes", value="")
        MntSweetProducts = st.text_input(label="Quantidade de Produto sendo Doces", value="")
        MntWines = st.text_input(label="Quantidade de Produto sendo Vinhos", value="")
        NumCatalogPurchases = st.text_input(label="Qtd Compras Catálogo", value="")
        NumDealsPurchases = st.text_input(label="Number Deals Purchases", value="")
        NumStorePurchases = st.text_input(label="Number Store Purchases", value="")
        NumWebPurchases = st.text_input(label="Qtd Compras Web", value="")
        NumWebVisitsMonth = st.text_input(label="Qtd Visitas Web por mês", value="")
        Recency = st.text_input(label="Recência", value="")
        Teenhome = st.text_input(label="Qtd Adolescentes em casa", value="")
        Time_Customer = st.text_input(label="Data de Compra", value="")

        with open('user_input.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header row (column names)
            writer.writerow(['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Age', 'Complain',	'Education', 'Income', 'Kidhome', 'Marital_Status',	'MntFishProducts', 'MntFruits',	'MntGoldProds',	'MntMeatProducts', 'MntSweetProducts', 'MntWines', 'NumCatalogPurchases', 'NumDealsPurchases', 'NumStorePurchases', 'NumWebPurchases', 'NumWebVisitsMonth', 'Recency', 'Teenhome', 'Time_Customer'])
            writer.writerow([AcceptedCmp1, AcceptedCmp2, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, Age, Complain, Education, Income, Kidhome, Marital_Status, MntFishProducts, MntFruits, MntGoldProds, MntMeatProducts, MntSweetProducts, MntWines, NumCatalogPurchases, NumDealsPurchases, NumStorePurchases, NumWebPurchases, NumWebVisitsMonth, Recency, Teenhome, Time_Customer])


        #carregamento do CSV
        Xtest = pd.read_csv('user_input.csv')

        #carregamento / instanciamento do modelo pkl
        tuned_lgbm = load_model('./pickle/pickle_et_pycaret')

        #predict do modelo
        ypred = predict_model(tuned_lgbm, data = Xtest, raw_score = True)

        with st.expander('Visualizar CSV carregado:', expanded = False):
            c1, _ = st.columns([2,4])
            qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:', 
                                    min_value = 5, 
                                    max_value = Xtest.shape[0], 
                                    step = 10,
                                    value = 5)
            st.dataframe(Xtest.head(qtd_linhas))

        with st.expander('Visualizar Predições:', expanded = True):
            c1, _, c2, c3 = st.columns([2,.5,1,1])
            treshold = c1.slider('Treshold (ponto de corte para considerar predição como True)',
                                min_value = 0.0,
                                max_value = 1.0,
                                step = .1,
                                value = .5)
            qtd_true = ypred.loc[ypred['prediction_score_1'] > treshold].shape[0]

            c2.metric('Qtd clientes True', value = qtd_true)
            c3.metric('Qtd clientes False', value = len(ypred) - qtd_true)
            
            def color_pred(val):
                color = 'green' if val > treshold else 'red'
                return f'background-color: {color}'

            tipo_view = st.radio('', ('Completo', 'Apenas predições'))
            if tipo_view == 'Completo':
                df_view = ypred.copy()
            else:
                df_view = pd.DataFrame(ypred.iloc[:,-1].copy())

            st.dataframe(df_view.style.applymap(color_pred, subset = ['prediction_score_1']))

            csv = df_view.to_csv(sep = ';', decimal = ',', index = True)
            st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
            st.download_button(label = 'Download CSV',
                            data = csv,
                            file_name = 'Predicoes.csv',
                            mime = 'text/csv')
