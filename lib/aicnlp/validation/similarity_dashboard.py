from jupyter_dash import JupyterDash
import dash
from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
import math
from pprint import pformat
import datetime
import random

from sklearn.metrics.pairwise import cosine_similarity


def labeled(label, elem, lsize="100px"):
    return html.Div([
        html.Label(label),
        elem
    ])


def get_layout(mgr):
    fig = px.imshow(
        np.random.random((10, 10)),
        color_continuous_scale='viridis',
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=0),
    )

    commonstyle = {"width": "100%", "border": "solid 1px", "padding": "3px"}

    dd_vec_op = mgr.get_dropdown_dicts()
    dd_vec_val = 'lsa_d200_e10'
    dd_mat_op = mgr.get_dropdown_dicts(dd_vec_val)
    dd_mat_val = 'RV'

    pac_vs_value = "p1 versus p2"
    if len(mgr.pac) > 1:
        pac_vs_value = mgr.pac.loc[0, "xml"] + " vs " + mgr.pac.loc[1, "xml"]


    nav_setting =  html.Div(children=[
        labeled("Document similarity method",
            dcc.Dropdown(
                options=dd_vec_op,
                value=dd_vec_val,
                id='dd-vec-method'
        )),
        labeled("Matrix similarity method",
            dcc.Dropdown(
                options=dd_mat_op,
                value=dd_mat_val,
                id='dd-mat-method'
        )),
        html.Hr(),
        labeled("First patient:",
            html.Div(children=[
                dcc.Slider(id="sl-pac-first", min=0, max=len(mgr.pac)-1, value=0,
                           tooltip={"placement": "left", "always_visible": True}),
            ],  style={"margin": "10px"}),
        ),
        dcc.Loading(
            html.Pre(id="text-Second-details", children="details",
                     style={"white-space": "pre-wrap"})
        ),
    ], style={**commonstyle,"padding": "10px", "height": "500px"})


    nav_vs = html.Div(children=[
        # html.Button('Show', id='pb-n3-refresh', n_clicks=0),
        dcc.Input(id="in-n3-p1", type="text", debounce=True, value=pac_vs_value,
                  style={"display": "inline-block", "width": "430px"}),
        labeled("n-th simmilar:",
            html.Div(children=[
                dcc.Slider(id="sl-pac-sim-index", min=0, max=len(mgr.pac)-1, value=0,
                           tooltip={"placement": "left", "always_visible": True}),
            ],  style={"margin": "10px"}),
        ),
        dcc.Loading(id="loading-n3", children=[
            dcc.Graph(id='graph-n3', figure=fig),
        ]),
    ], style={**commonstyle, "height": "450px"})


    nav_pvp_view = html.Div(children=[
        html.Div(children=[
            dcc.Loading(id="loading-n4-1", children=[
                html.H3(id="text-p1", children='P1'),
                html.Pre(id="text-p1-body",
                         style={"white-space": "pre-wrap", "overflow-y": "scroll", "height": "380px"}),
            ])
        ], style={"width": "49%",'float': 'left'}),
        html.Div(children=[
            dcc.Loading(id="loading-n4-2", children=[
                html.H3(id="text-p2", children='P2'),
                html.Pre(id="text-p2-body",
                         style={"white-space": "pre-wrap",  "overflow-y": "scroll", "height": "380px"}),
            ])
        ], style={"width": "49%",'float': 'right'})

    ], style={**commonstyle, "height": "450px", "padding": "5px"})


    nav_scatter = html.Div(children=[
        html.H3(id="text-scatter-1", children='Similar patients to ???'),
        dcc.Loading(id="loading-g_scatter", children=[
            dcc.Graph(id='graph-scatter'),
        ]),
    ], style={**commonstyle, "height": "500px"})


    nav_new =  dbc.Container(html.Div([
        dbc.Row([
            dbc.Col(nav_setting, width=3, align="stretch", style={"padding": "0"}),
            dbc.Col(nav_scatter, width=9, align="stretch", style={"padding": "0"})
        ]),
        dbc.Row([
            dbc.Col(nav_vs, width=3, align="stretch", style={"padding": "0"}),
            dbc.Col(nav_pvp_view, width=9, align="stretch", style={"padding": "0"})
        ]),
    ]), fluid=True)

    return nav_new


replacement_groups = [
    "0123456789",
    "aeiyou",
    "aeiyou".upper(),
    "qwrtpsdfghjklzxcvbnměščřžýáíéůúďťň",
    "qwrtpsdfghjklzxcvbnměščřžýáíéůúďťň".upper(),
]
def replace_char(c):
    for group in replacement_groups:
        if c in group:
            return random.choice(group)
    return c


def make_safe(text):
    return "".join(replace_char(c) for c in text)


###################################################
##########          Callbacks        ##############
###################################################
def add_callbacks(app, mgr):

    def get_patient_ids(patients):
        l = patients.split()
        if len(l) != 3:
            return None, None

        rowPatient, _, colPatient = l
        tr = mgr.tr["xml", "pid"]
        if rowPatient not in tr:
            raise KeyError(f"Could not find {repr(rowPatient)}")
        if colPatient not in tr:
            raise KeyError(f"Could not find {repr(colPatient)}")

        rowPatient = tr[rowPatient]
        colPatient = tr[colPatient]
        return rowPatient, colPatient

    @app.callback(
        # Output('text-patients', 'children'),
        Output('graph-n3', 'figure'),
        Output("in-n3-p1", "value"),
        Output('sl-pac-sim-index', 'value'),
        Input('dd-vec-method', 'value'),
        Input('dd-mat-method', 'value'),
        Input('in-n3-p1', 'value'),
        Input('sl-pac-sim-index', 'value'),
        Input('graph-scatter', 'clickData'),
    )
    def update_g3(vec_method, mat_method, patients, pac_sim_idx, scatter_click):
        ctx = dash.callback_context
        if not ctx.triggered:
            trigger_id = None
        else:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        nullfig = px.imshow([[]], color_continuous_scale='viridis',
                            range_color=[0,1], height=300)

        rowPatient, colPatient = get_patient_ids(patients)
        if rowPatient is None:
            return nullfig, patients, pac_sim_idx

        # in-n3-p1
        # sl-pac-sim-index
        row = mgr.get_sims(vec_method, mat_method)[rowPatient, :].copy()
        row[rowPatient] = 1
        sorted_others = row.argsort()[::-1]

        if trigger_id == "sl-pac-sim-index":
            colPatient = sorted_others[pac_sim_idx]
        elif trigger_id == "graph-scatter":
            colPatient = scatter_click["points"][0]['pointIndex']
            pac_sim_idx = sorted_others.tolist().index(colPatient)
        else:
            pac_sim_idx = sorted_others.tolist().index(colPatient)

        patients = mgr.tr["pid", "xml"][rowPatient] + " vs " + mgr.tr["pid", "xml"][colPatient]

        try:
            ar = np.vstack(mgr.get_pac_vectors(vec_method, rowPatient, key="pid"))
            ac = np.vstack(mgr.get_pac_vectors(vec_method, colPatient, key="pid"))
        except KeyError:
            return nullfig, patients, pac_sim_idx

        sim = cosine_similarity(ar, ac)

        fig = px.imshow(sim, color_continuous_scale='viridis',
                        range_color=[0,1], height=300)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0))

        return fig, patients, pac_sim_idx


    @app.callback(
        Output('text-p1', 'children'),
        Output('text-p1-body', 'children'),
        Output('text-p2', 'children'),
        Output('text-p2-body', 'children'),
        Input('graph-n3', 'clickData'),
        # Input('pb-n3-refresh', 'n_clicks'),
        State('in-n3-p1', 'value'),
        # State('in-n3-p2', 'value'),
        # State('text-patients', 'children'),
    )
    def graph3_clicked(data, patients):
        if data is None or "vs" not in patients:
            return "Nothing selected", "", "Nothing selected", ""

        colRecId = data["points"][0]["x"]
        rowRecId = data["points"][0]["y"]

        p1, _, p2 = patients.split()
        tr = mgr.tr["xml", "pid"]
        if p1 not in tr:
            print("Could not find", p1)
        if p2 not in tr:
            print("Could not find", p2)
        pid1 = tr[p1]
        pid2 = tr[p2]

        p1_text = mgr.rec.loc[:, pid1, rowRecId].iloc[0]["text"]
        p2_text = mgr.rec.loc[:, pid2, colRecId].iloc[0]["text"]

        if DO_LOREM:
            p1_text = make_safe(p1_text)
            p2_text = make_safe(p2_text)

        return f"{p1}: {rowRecId}", p1_text, f"{p2}: {colRecId}",  p2_text


    @app.callback(
        # Output('text-patients', 'children'),
        Output('graph-scatter', 'figure'),
        Output('text-scatter-1', 'children'),
        Output('text-Second-details', 'children'),
        Input('dd-vec-method', 'value'),
        Input('dd-mat-method', 'value'),
        Input('in-n3-p1', 'value'),
    )
    def update_scatter(vec_method, mat_method, patients):
        nullfig = px.imshow([[]], color_continuous_scale='viridis',
                    range_color=[0,1], height=300)
        n_patients = "???"

        rowPatient, colPatient = get_patient_ids(patients)
        if rowPatient is None:
            return nullfig, f"Similar patients to {row_xml}", ""

        row_xml = mgr.tr["pid", "xml"][rowPatient]
        pid = rowPatient
        n_patients = len(mgr.pac)

        selected = [1, 2, 2133, 4260, 4261]

        df = pd.DataFrame({
            "lsa": mgr.get_sims(vec_method, mat_method)[pid, :].copy(),
            "n_records": mgr.rec.groupby("pid")["text"].count().values,
            "pid": range(n_patients),
            "xml": [mgr.tr["pid", "xml"][i] for i in range(n_patients)],
            "id_pac": [mgr.tr["pid", "id_pac"][i] for i in range(n_patients)],
        })
        fig = px.scatter(
            df, x="lsa", y="n_records", marginal_x="histogram", marginal_y="histogram",
            hover_data=['pid', 'xml', 'id_pac'],
        )

        fig.add_shape(
            type='line',x0=df.loc[colPatient, "lsa"], y0=0,
            x1=df.loc[colPatient, "lsa"], y1=df.n_records.max(),
            line=dict(color='Red',), xref='x', yref='y'
        )


        details = str(df.loc[colPatient].to_dict())[1:-1]
        return fig, f"Similar patients to {row_xml}", details


DO_LOREM = False

def get_app(mgr, name="simdash", lorem=False):
    global DO_LOREM
    DO_LOREM = lorem
    external_stylesheets = [dbc.themes.BOOTSTRAP]

    app = JupyterDash(name, external_stylesheets=external_stylesheets)

    app.layout = get_layout(mgr)

    add_callbacks(app, mgr)

    return app
