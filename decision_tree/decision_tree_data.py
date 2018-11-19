import pandas as pd

def toy_data():
    data = pd.DataFrame({'age':['young', 'young', 'young', 'young', 'young', 'middle', 'middle', 'middle', 'middle', 'middle', 'old', 'old', 'old', 'old', 'old'],
                        'job':['no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no'],
                        'house':['no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'yes','yes', 'yes','yes', 'no', 'no', 'no'],
                        'debt': ['normal', 'good', 'good', 'normal', 'normal', 'normal', 'good', 'good', 'excellent', 'excellent', 'excellent', 'good', 'good', 'excellent', 'normal'],
                        'label': ['no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']})
    return data                    
