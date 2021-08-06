from utils import save_results
import pandas as pd
import csv
def save_result(csv, dict):
    '''
    This function is used to save the conditions and results of training the DNN in a csv file
    Args:
        csv (str): The name of the csv file. Must be in the format 'XXX.csv'
        dict (dict): The conditions and results of training in the form of a dictionary

    Returns:
        None
    '''
    df = pd.read_csv(csv, index_col=0)
    df = df.append(dict, ignore_index=True)
    df.to_csv(csv)

dict = {'Organ': 'myorgan'}

header = ['Organ']


with open('mresults.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)


save_result('mresults.csv', dict)