import dash
from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from validity_index import one_hot_encode, calculate_xie_beni_index, calculate_partition_coefficient, calculate_partition_entropy, calculate_external_criteria
import pandas as pd
from fcm import fcm, init_membership_matrix
from ssfcm import ssfcm
from esfcm import esfcm
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
app.title = "Fuzzy c-Means Clustering"
server = app.server

input_style = {
    "width": "100%",
    "padding": "5px",
    "fontSize": "145x",
    "border": "1px solid #ced4da",
    "borderRadius": "4px",
    "boxShadow": "inset 0 1px 2px rgba(0, 0, 0, 0.075)",
}


app.layout = html.Div(
    [
        html.H1(
            children="Fuzzy c-Means Clustering",
            className="title",
            style={"textAlign": "center"},
        ),
        html.Div(
            style={
                'display': 'flex',  # Use flexbox for layout
                'flexDirection': 'row',  # Arrange children in a row
                'height': '100vh',  # Full viewport height
            },
            children=[
                # Left Panel
                html.Div(
                    style={
                        'flex': '1',  # Take up one part of the horizontal space
                        'backgroundColor': '#f8f9fa',  # Light grey background
                        'padding': '20px',  # Inner spacing
                        'borderRight': '2px solid #dee2e6',  # Add a dividing border
                    },
                    children=[
                        html.H4("Configuration"),
                        # html.P("This is the left panel where you can place controls."),
                        # Example of controls
                        html.Label("Dataset:"),
                        dcc.Dropdown(
                            id="dataset-dropdown",  # Unique ID for callback reference
                            options=[
                                {'label': 'Iris', 'value': 'iris'},
                                {'label': 'Wine', 'value': 'wine'},
                            ],
                            placeholder="Choose a dataset",  # Placeholder text
                        ),
                        html.Label("Algorithm:"),
                        dcc.Dropdown(
                            id="algorithm-dropdown",  # Unique ID for callback reference
                            options=[
                                {'label': 'Unsupervised FCM', 'value': 'fcm'},
                                {'label': 'Semi-Supervised Standard FCM', 'value': 'ssfcm'},
                                {'label': 'Semi-Supervised Entropy Regularized FCM', 'value': 'esfcm'},
                            ],
                            placeholder="Choose an algorithm",  # Placeholder text
                        ),
                        html.Label("Number of Clusters:"),
                        dcc.Dropdown(
                            id="nclusters-dropdown",  # Unique ID for callback reference
                            options=[{'label': i, 'value': i} for i in range(2, 11)],
                            placeholder="Choose the number of clusters",  # Placeholder text
                        ),
                        html.Label("Max iterations:"),
                        dcc.Dropdown(
                            id="max-iter-dropdown",  # Unique ID for callback reference
                            options=[{'label': i, 'value': i} for i in [100, 300, 500, 1000, 2000]],
                            placeholder="Choose the maximum number of iterations",  # Placeholder text
                        ),
                        html.Label("Convergence threshold (\u03B5):"),
                        dcc.Dropdown(
                            id="eps-dropdown",  # Unique ID for callback reference
                            options=[{'label': i, 'value': i} for i in [1e-2, 1e-3, 1e-4, 1e-5]],
                            placeholder="Choose \u03B5",  # Placeholder text
                        ),
                        html.Div(
                            id="dynamic-hyperparameter-container",
                            children=[
                                html.Div(
                                    id="lambda-container",
                                    children=[
                                        html.Label("Regularization weight (\u03BB):"),
                                        dcc.Input(
                                            id="lambda-input",
                                            type="number",
                                            placeholder="Enter \u03BB",
                                            step = 0.01,
                                            style=input_style
                                        ),

                                    ],
                                    style={"display": "none"}
                                ),
                                html.Div(
                                    id="m-container",
                                    children=[
                                        html.Label("Fuzzification coefficient (m):"),
                                        dcc.Input(
                                                id="m-input",
                                                type="number",
                                                placeholder="Enter m",
                                                step = 0.01,
                                                min=1,
                                                style=input_style,
                                            )
                                    ],
                                    style={"display": "none"}
                                ),
                                html.Div(
                                    id="normalize-container",
                                    children=[
                                        html.Label("Normalization:"),
                                        dcc.Dropdown(
                                            id="normalize-dropdown",
                                            options=[
                                                {'label': 'Yes', 'value': 'yes'},
                                                {'label': 'No', 'value': 'no'},
                                            ],
                                            # placeholder="Choose an algorithm", 
                                        )
                                    ],
                                    style={"display": "none"}
                                ),
                                html.Div(
                                    id="init-container",
                                    children=[
                                        html.Label("Initialization Method:"),
                                        dcc.Dropdown(
                                            id="init-dropdown",
                                            options=[
                                                {'label': 'Semi-supervised', 'value': 'ss'},
                                                {'label': 'Random', 'value': 'random'},
                                            ],
                                            placeholder="Choose initialization method", 
                                        )
                                    ],
                                    style={"display": "none"}
                                ),
                            ]
                        ),
                        html.Div(
                            style = {
                                'align-items': 'center',
                                'display': 'flex',
                                'justify-content': 'center',
                                'flexDirection': 'column',
                                },
                            children = [
                            html.Button(
                                "Run Algorithm",
                                id="run-button",
                                n_clicks=0,
                                style={
                                    "marginTop": "20px",
                                    "backgroundColor": "#4CAF50",
                                    "color": "white",
                                    "border": "none",
                                    "padding": "10px 20px",
                                    "textAlign": "center",
                                    # "textDecoration": "none",
                                    # "display": "inline-block",
                                    "fontSize": "16px",
                                    "borderRadius": "8px",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Div(id="warning-message", style={"color": "red", "marginTop": "10px"}),]
                        ),
                        
                    ],
                    
                ),
                
                # Right Panel
                html.Div(
                    style={
                        'flex': '3',  # Take up three parts of the horizontal space
                        'backgroundColor': '#ffffff',  # White background
                        'padding': '20px',  # Inner spacing
                        "overflowY": "auto", 
                        "height": "1000px", 
                        "maxHeight": "100%"
                    },
                    children=[
                        html.H4("Output"),
                        html.Div(id="output-container"
                            )
                    ],
                ),
            ]
        )
    ]
)

def get_input(dataset: str, algorithm: str, n_clusters: int):
    U_supervised = None
    df = pd.read_csv(f"data\\{dataset}.csv")
    input_df = df.drop("class", axis=1)
    X = np.array(input_df)
    U_true = one_hot_encode(df["class"])
    if algorithm in ["ssfcm", "esfcm"]:
        U_supervised = np.array(pd.read_csv(f"data\\{dataset}_supervised.csv"))
        N, c = U_supervised.shape
        if n_clusters < c:
            U_supervised = U_supervised[:, :n_clusters]
        elif n_clusters > c:
            U_supervised = np.hstack((U_supervised, init_membership_matrix(N, n_clusters-c, 0.4)))
    return X, U_true, U_supervised, input_df
        
# Callback to show the dynamic hyperparameter based on the algorithm
@app.callback(
    [
        Output("lambda-container", "style"),
        Output("m-container", "style"),
        Output("normalize-container", "style"),
        Output("init-container", "style"),
    ],
    Input("algorithm-dropdown", "value")
)
def toggle_hyperparameters(algorithm):
    if algorithm == "esfcm":
        return {"display": "block"}, {"display": "none"}, {"display": "block"}, {"display": "block"}
    elif algorithm == "ssfcm":
        return {"display": "none"}, {"display": "block"}, {"display": "block"}, {"display": "block"}
    elif algorithm =="fcm":
        return {"display": "none"}, {"display": "block"}, {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "none"}


# Callback to handle the Run Algorithm button click
@app.callback(
    [Output("warning-message", "children"), 
     Output("output-container", "children"),
    ],
    Input("run-button", "n_clicks"),
    [
        State("dataset-dropdown", "value"),
        State("algorithm-dropdown", "value"),
        State("nclusters-dropdown", "value"),
        State("max-iter-dropdown", "value"),
        State("eps-dropdown", "value"),
        State("lambda-input", "value"),
        State("m-input", "value"),
        State("normalize-dropdown", "value"),
        State("init-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def run_algorithm(n_clicks, dataset, algorithm, n_clusters, max_iter, eps, lambda_, m, normalize, init):
    if not all([dataset, algorithm, n_clusters, max_iter, eps]):
        return "Please fill in all required configurations.", ""
    
    if algorithm == "esfcm" and not all([lambda_, normalize, init]):
        return "Please fill in all required configurations.", ""

    if algorithm == "ssfcm" and not all([m, normalize, init]):
        return "Please fill in all required configurations.", ""
    
    if algorithm == "fcm" and not all([m, normalize]):
        return "Please fill in all required configurations.", ""
    
    if init == "ss":
        init = True
    elif init == "random":
        init = False

    X, U_true, U_supervised, input_df = get_input(dataset, algorithm, n_clusters)

    if normalize=="yes":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    U, V = None, None
    if algorithm == "fcm":
        U, V = fcm(X, n_clusters, m, max_iter, eps)
    elif algorithm == "ssfcm":
        U, V = ssfcm(X, n_clusters, U_supervised, m, max_iter, eps, init)
    elif algorithm == "esfcm":
        U, V = esfcm(X, n_clusters, U_supervised, lambda_, max_iter, eps, init)

    xie_beni_index = round(calculate_xie_beni_index(U, V, X), 4)
    partition_coef = round(calculate_partition_coefficient(U), 4)
    partition_entropy = round(calculate_partition_entropy(U), 4)
    FRI, FARI, FJI = calculate_external_criteria(U, U_true)
    FRI, FARI, FJI = round(FRI, 4), round(FARI, 4), round(FJI, 4)
    input_table = dash_table.DataTable(
        id='input-table',
        columns=[
            {"name": col, "id": col} for col in input_df.columns
        ],
        data=input_df.to_dict("records"),  # Provide data for the table
        style_table={'overflowX': 'auto',"width": "700px", "maxWidth":"100%"},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'whiteSpace': 'normal',
            "fontSize": "14px",
            "fontFamily": "Arial, sans-serif",
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )

    membership_df = pd.DataFrame(np.round(U, 4), columns=[str(i) for i in range(n_clusters)])
    membership_table = dash_table.DataTable(
        id="membership-table",
        columns=[{"name": col, "id": col} for col in membership_df.columns],
        data=membership_df.to_dict("records"),
        style_table={"overflowX": "auto", "width": "300px", "maxWidth": "100%"},
        style_cell={
            "textAlign": "left",
            "padding": "10px",
            "fontSize": "14px",
            "fontFamily": "Arial, sans-serif",
            "whiteSpace": "normal",
        },
        style_header={
            "backgroundColor": "rgb(230, 230, 230)",
            "fontWeight": "bold",
        },
    )

    tables_containter = html.Div(
        style={
            "display": "flex", 
            "gap": "50px",
            "overflowY": "auto", 
            "height": "450px", 
            "maxHeight": "100%"
            },
        children=[
            html.Div(
                style={"flex": "1"},
                children=[
                    html.Label("Input:"),
                    input_table
                ]
                ),
            html.Div(
                style={"flex": "1"},
                children=[
                    html.Label("Membership:"),
                    membership_table
                ]
                ),
        ],
    )

    relative_indices_df = pd.DataFrame(["Xie-Beni Index:", xie_beni_index, "Partition Coefficient:", partition_coef, "Partition Entropy:", partition_entropy]).swapaxes("index", "columns")
    relative_indices_df.columns = [str(i) for i in range(6)]
    external_indices_df = pd.DataFrame(["Rand Index:", FRI, "Adjusted Rand Index:", FARI, "Jaccard Index:", FJI]).swapaxes("index", "columns")
    external_indices_df.columns = [str(i) for i in range(6)]
    col1 = [{"name": "", "id": col} for col in relative_indices_df.columns]
    col1[0]["name"] = "Relative validity indices:"
    col2 = [{"name": "", "id": col} for col in external_indices_df.columns]
    col2[0]["name"] = "External validity indices:"
    indices_container = html.Div(
        [
            dash_table.DataTable(
                columns=col1,
                data=relative_indices_df.to_dict("records"),
                style_cell={
                    "border": "none",
                    "textAlign": "left",
                    "padding": "10px",
                    "fontSize": "14px",
                    "fontFamily": "Arial, sans-serif",
                    "whiteSpace": "normal",
                    "width":"150px"
                },
                # style_header={"fontWeight": "bold"}
            ),
            dash_table.DataTable(
                columns=col2,
                data=external_indices_df.to_dict("records"),
                style_cell={
                    "border": "none",
                    "textAlign": "left",
                    "padding": "10px",
                    "fontSize": "14px",
                    "fontFamily": "Arial, sans-serif",
                    "whiteSpace": "normal",
                    "width":"150px"
                },
                # style_header={"fontWeight": "bold"}
            )
        ]
    )

    output = html.Div(
        [
            tables_containter,
            indices_container,
        ]
    )

    return "", output

if __name__ == "__main__":
    app.run(debug=True)



