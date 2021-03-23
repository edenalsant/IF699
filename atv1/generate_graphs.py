import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def generate_ratio(results):
    hit_rate = results['hit_rate']
    execution_time = results['execution_time']
    ratio = []
    r = []
    for i in range(len(hit_rate)):
        r.append(hit_rate[i][0]/execution_time[i][0])
        r.append(hit_rate[i][1]/execution_time[i][1])
        ratio.append(r)
        r = []

    return ratio

def graph(alg_name, data_set_name, results):

    results_df = pd.DataFrame(results)
    print("results: ")
    print(results_df)

    data = {
        'k': [1,3]
    }

    ratio = generate_ratio(results)

    data['knn_simples'] = ratio[0]
    data['lvq1_182'] = ratio[1]
    data['lvq1_37'] = ratio[2]
    data['lvq2.1_182'] = ratio[3]
    data['lvq2.1_37'] = ratio[4]
    data['lvq3_182'] = ratio[5]
    data['lvq3_37'] = ratio[6]

    knn_data = pd.DataFrame(data)
    print(knn_data)
    fig = make_subplots(rows=1, cols=1,
    specs=[[{"type": "Scatter"}]],
    subplot_titles=("Hit Rate/Execution Time"))

    
    fig.add_trace(go.Scatter(x=data['k'], y = data['knn_simples'], name='Simples', mode = 'lines+markers'), col = 1, row = 1) 
    fig.add_trace(go.Scatter(x=data['k'], y = data['lvq1_182'], name='LVQ1 1459 prot', mode = 'lines+markers'), col = 1, row = 1)
    fig.add_trace(go.Scatter(x=data['k'], y = data['lvq2.1_182'], name='LVQ2.1 1459 prot', mode = 'lines+markers'), col = 1, row = 1)
    fig.add_trace(go.Scatter(x=data['k'], y = data['lvq3_182'], name='LVQ3 1459 prot', mode = 'lines+markers'), col = 1, row = 1)
    fig.add_trace(go.Scatter(x=data['k'], y = data['lvq1_37'], name='LVQ1 296 prot', mode = 'lines+markers'), col = 1, row = 1)
    fig.add_trace(go.Scatter(x=data['k'], y = data['lvq2.1_37'], name='LVQ2.1 296 prot', mode = 'lines+markers'), col = 1, row = 1)
    fig.add_trace(go.Scatter(x=data['k'], y = data['lvq3_37'], name='LVQ3 296 prot', mode = 'lines+markers'), col = 1, row = 1)



    fig.update_layout(title_text='{} para dataset {}'.format(alg_name, data_set_name))
    fig.update_yaxes(title_text="Ratio")
    fig.update_layout(height=1000, width=1000)
    return fig.show()


results = {
    "hit_rate": [[80.58775510204082, 84.59591836734694],[92.83625730994152,92.83625730994152],[86.66666666666666, 95.0],[92.83625730994152,92.83625730994152],[92.83625730994152,92.83625730994152],[92.83625730994152,92.83625730994152],[92.83625730994152,92.83625730994152]],
    "execution_time": [[105.08503341674805, 91.37352108955383],[10.92712116241455,10.85326886177063],[0.573183536529541, 0.5560035705566406],[10.908060073852539,10.860605001449585],[12.967845916748047, 12.909904479980469],[10.769865036010742,10.72603440284729],[13.117284536361694, 13.431249618530273]]
}

graph("KNN", "CM1", results)
