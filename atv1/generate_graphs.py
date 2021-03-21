import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def generate_graph(alg_name, data_set_name, execution_time, hit_rate):
    knn_data = pd.DataFrame({
    'k': [1,2,3,5,7,9,11,13,15],
    'execution_time': execution_time,
    'hit_rate': hit_rate
    })

    x = knn_data.k

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=x,
        y=knn_data['hit_rate'],
        line=dict(color='mediumseagreen', width=1),
        name="Taxa de acerto"
        ))

    fig.add_trace(go.Scatter(
        x=x,
        y=knn_data['execution_time'],
        line=dict(color='fuchsia', width=1),
        #mode='lines',
        name="Tempo de Execução"
        ),
        secondary_y=True
    )

    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = knn_data['k']
        )
    )

    fig.update_layout(title_text='{} para data set {}'.format(alg_name, data_set_name))
    fig.update_yaxes(title_text="Taxa de acerto (%)", secondary_y=False)
    fig.update_yaxes(title_text="Tempo de Execução (s)", secondary_y=True)

    fig.show()

# CM1 
data_set_name = 'CM1'
# KNN simples
alg_name = 'KNN simples'
execution_time = [105.08503341674805, 99.70476961135864, 91.37352108955383, 89.40125632286072, 106.92617130279541, 93.50601959228516, 92.14104890823364, 93.0115852355957, 98.81660509109497]
hit_rate = [80.58775510204082, 80.58775510204082, 84.59591836734694, 86.6, 88.4, 89.00408163265305, 89.00408163265305, 89.60408163265306, 89.80408163265307]

generate_graph(alg_name, data_set_name, execution_time, hit_rate)


# KNN pesado
alg_name = 'KNN com pesos'
execution_time = [1643.3551275730133, 1614.9473490715027, 1472.3689367771149, 1478.1312658786774, 1474.0493786334991, 1464.2704298496246, 2053.48180603981, 2256.4391207695007, 1881.6942949295044]
hit_rate = [74.6872037914692, 75.49334236064092, 78.15007898894154, 79.00383660573235, 79.47731888964117, 79.76167907921462, 79.90453622207177, 80.09388399909727,  80.56804333107651]

generate_graph(alg_name, data_set_name, execution_time, hit_rate)

# KNN adaptativo
alg_name = 'KNN adaptativo'
execution_time = [375.5588583946228, 273.25959062576294,252.3770613670349, 252.07808351516724, 251.39574575424194 ,252.0083122253418, 251.879141330719, 251.85869145393372,252.06518578529358]
hit_rate = [88.80408163265307,88.80408163265307,88.80408163265307,88.80408163265307,88.80408163265307,88.80408163265307,88.80408163265307,88.80408163265307,88.80408163265307]

generate_graph(alg_name, data_set_name, execution_time, hit_rate)

#KC1
data_set_name = 'KC1'

#KNN Simples
alg_name = 'KNN simples'
execution_time = [2410.320837497711, 2245.502151966095, 1721.5718579292297, 1455.2546923160553,1622.2996757030487, 1728.9506599903107, 1438.1285490989685,1414.6580624580383, 1422.493162870407 ]
hit_rate = [74.6872037914692, 74.6872037914692, 79.52471225457006, 80.70977206048296, 80.94651320243737, 80.89911983750845, 81.65831640713157, 81.75310313698938, 82.03746332656286]

generate_graph(alg_name, data_set_name, execution_time, hit_rate)

#KNN pesado
alg_name = 'KNN com pesos'
execution_time = [1643.3551275730133,1614.9473490715027,1472.3689367771149,1478.1312658786774, 1474.0493786334991,1464.2704298496246,2053.48180603981,2256.4391207695007,1881.6942949295044]
hit_rate = [74.6872037914692, 75.49334236064092, 78.15007898894154, 79.00383660573235,79.47731888964117, 79.76167907921462, 79.90453622207177,80.09388399909727,80.56804333107651 ]

generate_graph(alg_name, data_set_name, execution_time, hit_rate)

#KNN Adaptativo
alg_name = 'KNN adaptativo'
execution_time = [6473.203100442886, 5984.097929239273, 5827.384940624237, 5796.26749920845, 5792.562381029129, 5807.304266214371, 5782.0247797966, 5799.370792865753, 5953.443441867828]
hit_rate = [82.55879034078086, 82.55879034078086,82.55879034078086,82.55879034078086,82.55879034078086,82.55879034078086,82.55879034078086,82.55879034078086,82.55879034078086]

generate_graph(alg_name, data_set_name, execution_time, hit_rate)
