import pandas as pd
import plotly.graph_objs as go
import tempfile

def graph_2d_opt(stats_params_backtest, param_1, param_2, obj_metric, ticker, add_text=False):
    heatmap_data = pd.pivot_table(stats_params_backtest, values=obj_metric, index=param_2, columns=param_1)
    
    # Crear la figura del heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Viridis',
        colorbar=dict(title=obj_metric)
    ))
    
    if add_text:
        # Añadir los valores numéricos en cada celda
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                fig.add_annotation(
                    text=f"{heatmap_data.values[i][j]:.2f}",  # Formato de dos decimales
                    x=heatmap_data.columns[j],
                    y=heatmap_data.index[i],
                    showarrow=False,
                    font=dict(size=10, color='#242323')  # Tamaño de letra ajustado
                )
        
    # Actualizar el layout de la figura
    fig.update_layout(
        title={
            'text': f'{obj_metric} - {param_1} vs {param_2} for {ticker}',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title=param_1,
        yaxis_title=param_2,
        width=1200,
        height=600,
        plot_bgcolor='#161616',
        paper_bgcolor='#161616',
        font=dict(color='#b2b2b2')
    )
    
    # Actualizar el hovertemplate
    fig.update_traces(
        hovertemplate=f"{param_1}: %{{x}}<br>{param_2}: %{{y}}<br>{obj_metric}: %{{z}}<extra></extra>"
    )
    
    # Guardar el gráfico en un archivo temporal
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_file:
        temp_path = temp_file.name
        fig.write_html(temp_path, auto_open=True)