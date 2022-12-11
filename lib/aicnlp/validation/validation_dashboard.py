from jupyter_dash import JupyterDash
import dash
from dash import Dash, html, dcc, callback_context
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math
from pprint import pformat
import datetime
import random

from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)
import aicnlp
PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"


from aicnlp.similarity.matsim import mms
from aicnlp.similarity.matsim import rv2
from aicnlp.similarity.matsim import eds


def labeled(label, elem, lsize="100px"):
    return html.Div([
        html.Label(label),
        elem
    ])


def get_layout(mgr, mean_annotations, all_predictions, all_correlations):
    fig = px.imshow(
        np.random.random((10, 10)),
        color_continuous_scale='viridis',
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=0),
    )

    commonstyle = {"width": "100%", "border": "solid 1px", "padding": "3px"}

    matmethods = list(set(x[0] for x in all_predictions.keys()))
    matmethod = "Rmms"
    vecmethods = list(set(set(x[1] for x in all_predictions.keys())))
    vecmethod = "Vrbc200"

    some_cor = next(iter(all_correlations.values()))
    some_pred = next(iter(all_predictions.values()))

    pivots = some_cor["pivot"].unique()
    pivot = pivots[0]

    categories = some_pred["cat"].unique()
    category = "all"

    global glob_vectors
    glob_vectors = None

    nav_setting =  html.Div(children=[
        labeled("Vectorization method",
            dcc.Dropdown(
                options=vecmethods,
                value=vecmethod,
                id='dd-vec-method'
        )),
        # labeled("Matrix similarity method",
        #     dcc.Dropdown(
        #         options=matmethods,
        #         value=matmethod,
        #         id='dd-mat-method'
        # )),
        labeled("Category",
            dcc.Dropdown(
                options=categories,
                value=category,
                id='dd-category'
        )),
        labeled("Pivot",
            dcc.Dropdown(
                options=pivots,
                value=pivot,
                id='dd-pivot'
        )),
        html.Hr(),
        dcc.Loading(
            html.Pre(id="text-vectors", children="details",
                     style={"white-space": "pre-wrap"})
        ),
        dcc.Loading(
            html.Pre(id="text-plotstat", children="",
                     style={"white-space": "pre-wrap"})
        ),
    ], style={**commonstyle,"padding": "10px", "height": "500px"})

    nav_relevants = html.Div(children=[
        html.H3(id="text-simmats-", children='Patients relevant to pivot'),
        dcc.Loading(id="loading-g_simmats", children=[
            dcc.Graph(id='graph-simmats'),
        ]),
    ], style={**commonstyle, "height": "500px"})

    nav_pvp_stat = html.Div(children=[
        dcc.Loading(
            html.Pre(id="text-pvpstat", children="",
                     style={"white-space": "pre-wrap"})
        ),
    ])

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


    nav_new =  dbc.Container(html.Div([
        dbc.Row([
            dbc.Col(nav_setting, width=2, align="stretch", style={"padding": "0"}),
            dbc.Col(nav_relevants, width=10, align="stretch", style={"padding": "0"})
        ]),
        dbc.Row([
            dbc.Col(nav_pvp_stat, width=2, align="stretch", style={"padding": "0"}),
            dbc.Col(nav_pvp_view, width=10, align="stretch", style={"padding": "0"})
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
def add_callbacks(app, mgr, mean_annotations, all_predictions, all_correlations):
    ma = mean_annotations.set_index(["pivot", "proxy", "cat"])

    some_cor = next(iter(all_correlations.values()))
    some_pred = next(iter(all_predictions.values()))

    @app.callback(
        Output('text-vectors', 'children'),
        Input('dd-vec-method', 'value'),
        Input('dd-category', 'value'),
    )
    def update_vectors(vecmethod, cat):
        if cat == "all":
            cat = "Fall"
        else:
            cat = f"Fr{cat}"
        global glob_vectors
        filename = f"{PACSIM_DATA}/2/{vecmethod}-{cat}.feather"
        glob_vectors = pd.read_feather(filename)
        return "\n".join([filename, vecmethod, cat])


    def get_one_row(pivot, cat):
        proxies = some_pred.query(f"pivot == '{pivot}'")["proxy"].unique()
        msg = []

        pivot_pid = mgr.tr["id_pac", "pid"][str(pivot)]
        pivot_vecs = glob_vectors.query(f"pid == {pivot_pid}")["vec"]
        if len(pivot_vecs) == 0:
            pivot_vecs = [glob_vectors["vec"].iloc[0]]
            msg.append("bad pivot")
        pivot_mat = np.vstack(pivot_vecs)

        out = {}
        for col, proxy in enumerate(proxies):
            proxy_pid = mgr.tr["id_pac", "pid"][str(proxy)]
            proxy_vecs = glob_vectors.query(f"pid == {proxy_pid}")["vec"]
            if len(proxy_vecs) == 0:
                proxy_vecs = [glob_vectors["vec"].iloc[0]]
                msg.append(f"bad {proxy}")
            proxy_mat = np.vstack(proxy_vecs)
            # out[f"Patient {col+1}"] = (pivot_mat, proxy_mat)
            if cat == "Fall":
                annotation = 1
            else:
                annotation = ma.loc[pivot, proxy, cat[2:]]["value"]
            out[f"{proxy}"] = (pivot_mat, proxy_mat, annotation)
        # print(proxies)
        return out, "\n".join(msg)


    def calculate_sims(p1, p2):
        results = []

        results.append(1+rv2.rv2(p1, p2))
        results.append(mms.mms(p1, p2))
        edssim, path = eds.eds(p1, p2, return_path=True)
        results.append(edssim)

        px = [pos[1] for pos in path]
        py = [pos[0] for pos in path]

        return mms.cosine_similarity(p1, p2), results, (px, py)

    def get_all_sims(mats):
        values = []
        new_mats = []
        paths = []
        names = []
        for name, (p1, p2, ann) in mats.items():
            new_mat, results, path = calculate_sims(p1, p2)
            values.append([ann] + results)
            new_mats.append(new_mat)
            paths.append(path)
            names.append(name)

        values = np.vstack(values).T
        # print(values)
        values = values / values.sum(axis=1)[:, np.newaxis]
        return zip(names, new_mats, values.T, paths)

    def make_plotshow(mats):
        ncol = len(mats)+1
        hcol = 5

        fig = make_subplots(rows=2, cols=ncol,
                            specs=[[{"type": "image"}]*ncol, [{"type": "table"}]*ncol] ,
                            column_widths=[hcol] + [(100-hcol)/(ncol-1)]*(ncol-1),
                            row_heights=[330, 120],
                            vertical_spacing=0,
                            shared_yaxes=True
                           )

        fig.add_table(
            cells=dict(
                values=[["ann", "rv2","mms", "eds"]],
                fill_color='white',),
            header=dict(
                values=[""],
                fill_color='white')
            ,row=2, col=1)


        for i, (name, new_mat, results, path) in enumerate(get_all_sims(mats)):
            results = [f"{r:.2f}" for r in results]
            lastim = px.imshow(new_mat, color_continuous_scale='viridis',aspect="equal", zmin=0, zmax=1)
            fig.add_trace(lastim.data[0],  row=1, col=i+2)
            fig.add_table(
                cells=dict(values=[results], fill_color='white',align=["center"]),
                header=dict(values=[name], fill_color='white'),
                row=2, col=i+2,
            )

        fig.layout.coloraxis = lastim.layout.coloraxis
        fig.update_yaxes(scaleanchor = "x3", showgrid=False, autorange='reversed')
        fig.update_xaxes(side="top", scaleanchor = "y3", showgrid=False,  autorange=True)
        fig.update_layout(height=300, width=900, autosize=True, plot_bgcolor="white", showlegend=False)
        fig.update_xaxes(showticklabels=False)
        fig.layout["yaxis2"].update(title="Pivot")
        fig.layout.update(
            title={"yref": "paper", "pad_b": 10, "x": 0.5, "y" : 1, "yanchor" : "bottom", "text": "Similarity scores for a pivot vs. 5 patients, d2v vectors"},
            margin_t=30, margin_b=0, margin_l=0,
        )
        fig.update_layout(height=450, width=1500, plot_bgcolor="white", showlegend=False)


        return fig


    @app.callback(
        Output('graph-simmats', 'figure'),
        Output('text-plotstat', 'children'),
        Input('text-vectors', 'children'),
        Input('dd-pivot', 'value'),
    )
    def update_relevants(vectors_text, pivot):
        filename, vecmethod, cat  = vectors_text.split("\n")
        mats, msg = get_one_row(pivot, cat)
        fig = make_plotshow(mats)
        return fig, msg


    def get_notes(id_pac):
        pid = mgr.tr["id_pac", "pid"][str(id_pac)]
        return glob_vectors.query(f"pid == {pid}")

    def get_note(id_pac, n):
        notes = get_notes(id_pac)
        if n >= len(notes):
            return "<<OUT OF BOUNDS>>"
        text = notes["text"].iloc[n]
        if DO_LOREM:
            text = make_safe(text)
        return text


    @app.callback(
        Output('text-pvpstat', 'children'),
        Output('text-p1', 'children'),
        Output('text-p1-body', 'children'),
        Output('text-p2', 'children'),
        Output('text-p2-body', 'children'),
        Input('graph-simmats', 'clickData'),
        Input('dd-pivot', 'value'),
    )
    def graph3_clicked(data, pivot):
        if data is None:
            return "", "Nothing selected", "", "Nothing selected", ""
        point = data["points"][0]
        pivot_nid = point["y"]
        proxy_nid = point["x"]
        proxy_index = point["curveNumber"] // 2
        proxies = some_pred.query(f"pivot == '{pivot}'")["proxy"].unique()
        proxy = proxies[proxy_index]

        p1 = pivot
        p2 = proxy
        p1_text = get_note(pivot, pivot_nid)
        p2_text = get_note(proxy, proxy_nid)

        return str(data), f"{p1}: note {pivot_nid}", p1_text, f"{p2}: note {proxy_nid}",  p2_text


DO_LOREM = False

def get_app(mgr, mean_annotations, all_predictions, all_correlations ,name=None, lorem=False):
    global DO_LOREM
    DO_LOREM = lorem
    external_stylesheets = [dbc.themes.BOOTSTRAP]
#     if name is None:
#         name = "srv_" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    app = JupyterDash(name, external_stylesheets=external_stylesheets)

    app.layout = get_layout(mgr, mean_annotations, all_predictions, all_correlations)

    add_callbacks(app, mgr, mean_annotations, all_predictions, all_correlations)

    return app
