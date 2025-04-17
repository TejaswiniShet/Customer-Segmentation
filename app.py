# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from flask import Flask, render_template, request

# Initialize Flask app
app = Flask(__name__)

# Step 1: Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Step 2: Preprocess the data
df = df.rename(columns={'Annual Income (k$)': 'Annual Income', 'Spending Score (1-100)': 'Spending Score (1-100)'})
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1}).astype(int)
X = df[['Age', 'Annual Income', 'Spending Score (1-100)']]
X = X.dropna()

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to generate elbow plot
def generate_elbow_plot(X_scaled):
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(x=list(range(1, 11)), y=inertia, mode='lines+markers', name='Inertia',
                                  line=dict(color='#00cc96', width=2), marker=dict(size=8, color='#00cc96')))
    elbow_fig.update_layout(
        title=dict(text='Elbow Method For Optimal k', x=0.5, xanchor='center', font=dict(size=20, color='#ffffff')),
        xaxis=dict(title=dict(text='Number of Clusters', font=dict(size=14, color='#ffffff')),
                   tickfont=dict(color='#ffffff')),
        yaxis=dict(title=dict(text='Inertia', font=dict(size=14, color='#ffffff')),
                   tickfont=dict(color='#ffffff')),
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0.1)',
        paper_bgcolor='rgba(0, 0, 0, 0.1)',
        showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return elbow_fig

# Function to generate plots based on selected attributes
def generate_cluster_plots(df, x_attr, y_attr, k):
    X = df[[x_attr, y_attr]].dropna()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled) + 1
    df.loc[X.index, 'Cluster'] = clusters

    # Scatter plot
    scatter_fig = go.Figure()
    for cluster in range(1, k + 1):
        cluster_data = df[df['Cluster'] == cluster]
        scatter_fig.add_trace(go.Scatter(x=cluster_data[x_attr], y=cluster_data[y_attr],
                                        mode='markers', name=f'Cluster {cluster}',
                                        marker=dict(size=10, opacity=0.7, line=dict(width=1, color='white'))))
    scatter_fig.update_layout(
        title=dict(text=f'Customer Segments ({x_attr} vs {y_attr})', x=0.5, xanchor='center', font=dict(size=20, color='#ffffff')),
        xaxis=dict(title=dict(text=x_attr, font=dict(size=14, color='#ffffff')),
                   tickfont=dict(color='#ffffff')),
        yaxis=dict(title=dict(text=y_attr, font=dict(size=14, color='#ffffff')),
                   tickfont=dict(color='#ffffff')),
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0.1)',
        paper_bgcolor='rgba(0, 0, 0, 0.1)',
        showlegend=True,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return scatter_fig, df

# Function to generate cluster summary
def generate_cluster_summary(df):
    summary = df.groupby('Cluster').agg({
        'Age': 'mean',
        'Annual Income': 'mean',
        'Spending Score (1-100)': 'mean',
        'Gender': lambda x: f"{(x == 0).mean():.2%} Male, {(x == 1).mean():.2%} Female"
    }).reset_index()
    # Custom HTML for styled table
    html = '<h3 style="text-align: center; color: #ffffff;">Cluster Summary</h3><table class="table table-striped" style="color: #ffffff; background-color: #2e2e2e; border: 2px solid #4B0082; border-radius: 5px; padding: 10px;">'
    html += '<tr><th>Cluster</th><th>Age</th><th>Annual Income</th><th>Spending Score (1-100)</th><th>Gender</th></tr>'
    for _, row in summary.iterrows():
        html += '<tr>'
        html += f'<td>{int(row["Cluster"])}</td>'
        html += f'<td>{row["Age"]:.6f}</td>'
        html += f'<td>{row["Annual Income"]:.6f}</td>'
        html += f'<td>{row["Spending Score (1-100)"]:.6f}</td>'
        html += f'<td>{row["Gender"]}</td>'
        html += '</tr>'
    html += '</table>'
    return html

# Initial elbow plot
elbow_fig = generate_elbow_plot(X_scaled)

@app.route('/', methods=['GET', 'POST'])
def display_output():
    global df, elbow_fig

    if request.method == 'POST':
        k = int(request.form.get('k'))
        x_attr = request.form.get('x_attr')
        y_attr = request.form.get('y_attr')

        # Generate new plots and summary
        scatter_fig, df = generate_cluster_plots(df, x_attr, y_attr, k)
        cluster_summary = generate_cluster_summary(df)
        return render_template('output.html',
                              elbow_plot=elbow_fig.to_html(include_plotlyjs='cdn'),
                              scatter_plot=scatter_fig.to_html(include_plotlyjs='cdn'),
                              cluster_summary=cluster_summary)

    # Initial render with only elbow plot and form
    return render_template('output.html',
                          elbow_plot=elbow_fig.to_html(include_plotlyjs='cdn'),
                          scatter_plot='',
                          cluster_summary='')

if __name__ == '__main__':
    app.run(debug=True)