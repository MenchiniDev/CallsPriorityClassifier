from functools import reduce
from tokenize import String
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.feature_selection import SelectKBest, f_classif

def make_association_rules():

    dataframe = pd.read_csv("../datasets/balancedDataset.csv")
    dataframe_selected = dataframe.copy()
    dataframe_selected = dataframe_selected.astype('object')

    for column in dataframe_selected.columns:
        transformed_column = dataframe_selected[column].apply(lambda x: f'{column} = {x}')
        dataframe_selected.loc[:, column] = transformed_column

    #apply Aprior
    te = TransactionEncoder()
    te_data = te.fit(dataframe_selected.values).transform(dataframe_selected.values)
    df = pd.DataFrame(te_data, columns=te.columns_)
    freq_itemset = apriori(df, min_support=0.09, use_colnames=True)
    
    freq_itemset.to_csv("../models/results/frequentItemset.csv", index=False)

    rules = association_rules(freq_itemset, metric="lift", min_threshold=0.1)
    data = pd.DataFrame(rules)
    output_rules = []
    for i in range(0, len(data)):
        antecedents = list(data.antecedents[i])
        consequents = list(data.consequents[i])

        for j in range(0, 2):
            print(antecedents, consequents)
            if not(f"priority = {j} " in antecedents) and not(f"priority = {j}" in consequents):
                ant = "IF "
                ant = reduce(lambda a, b: a + " AND " + b, antecedents) + " THEN "
                cons = reduce(lambda a, b: a + " AND " + b, consequents)
                rule = ant + cons
                output_rules.append(rule)

        with open("../models/results/associativeRules.txt", "a") as file:
            for rule in output_rules:
                file.write(rule + "\n")

# ar = make_association_rules()

