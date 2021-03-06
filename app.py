import pandas as pd
import streamlit as st

from src.config import APP_PAGE_HEADER
from src.utils import search_df

APP_PAGE_HEADER()


@st.cache(allow_output_mutation=True)
class LoadData:
    train: pd.DataFrame = pd.read_csv("data/train.csv")
    train = train.sample(frac=1).reset_index(drop=True)  # shuffle data

    test: pd.DataFrame = pd.read_csv("data/test.csv")
    titles: pd.DataFrame = pd.read_csv("data/titles.csv")

    # add code titles to train data
    merged = train.merge(titles, left_on="context", right_on="code")
    train_df = merged[['id', 'anchor', 'context', 'target', 'title', 'score']].copy()

    # add relations / edges for knowledge graph
    train_kg: pd.DataFrame = train_df.copy()
    train_kg['relation'] = train_kg['context'] + " || " + train_kg['title'] + " || " + train_kg['score'].astype(str)


class App:
    def __init__(self):
        self.data = LoadData()

    def run(self, debug=False):
        self.render_header(debug)
        self.render_body(debug)
        self.render_footer(debug)

    def render_header(self, *args, **kwargs):
        pass

    @staticmethod
    def render_body(*args, **kwargs):
        Helper().display_train_data()

        Helper().visualize()

    def render_footer(self, *args, **kwargs):
        pass


class Helper(App):

    def display_train_data(self):
        data = self.data.train_df

        st.write(f"> Train data `{data.shape[0]}` rows")
        filter_ = st.text_input("search phrases", "")
        if filter_:
            data = search_df(data, filter_)
        st.write(data)

    def visualize(self, *args, **kwargs):
        st.subheader("Visualize Phrases as a Network Graph")
        data = self.data.train_kg

        # filter data for visualization
        # -- sampling
        MAX_EDGES = 200
        sample = data[:MAX_EDGES]

        st1, st2 = st.columns(2)
        score = st1.selectbox("visualize phrases by similarity score", [""] + data["score"].unique().tolist())
        if score:
            sample = data[data["score"] == float(score)][:MAX_EDGES]

        filter_ = st2.text_input("search term to visualize matching phrases")
        if filter_:
            sample = search_df(data, filter_)[:MAX_EDGES]

        # create graph
        nodes = list(sample["anchor"].unique()) + list(sample["target"].unique())
        edges = [(h, t) for h, t in zip(sample["anchor"].tolist(), sample["target"].tolist())]
        labels = sample["relation"].tolist()
        edge_labels = dict(zip(edges, labels))

        # create PyVis network from the graph data
        self.pyvis_network(nodes, edge_labels)
        st.write(f"> the visualized sample size: {sample.shape[0]}")
        st.write(sample)

    def pyvis_network(self, nodes, edge_labels):

        from stvis import pv_static

        g = self.build_network(edge_labels, nodes)

        pv_static(g)

    @staticmethod
    @st.experimental_singleton
    def build_network(edge_labels, nodes):
        # src: https://stackoverflow.com/a/67279471/2839786
        from pyvis.network import Network
        g = Network(height="800px", width="1400px", heading="U.S. Patent Phrase/Context Network",
                    bgcolor="#bbbffz")  # notebook=True,
        for node in nodes:
            g.add_node(node)
        for e in edge_labels:
            n1, n2 = e[0], e[1]
            label = edge_labels[e]
            g.add_edge(n1, n2, title=label, show_edge_weights=True)  # weight 42
        return g


if __name__ == "__main__":
    app = App()
    app.run(debug=True)
