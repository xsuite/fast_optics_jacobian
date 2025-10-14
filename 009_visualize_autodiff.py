import jpviz
import xtrack as xt

import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt

# Build a simple ring
colour = 'gray'

# Prepare the lattice and plot the beam
# =====================================

# plt.style.use('../../latex.mplstyle')

# Create an environment
env = xt.Environment()

# Build a simple ring
env['mq1'] = 0.5
env['mq2'] = 0.5


line = env.new_line(components=[
    env.new('START', xt.Marker),
    env.new('d0.1', xt.Drift, length=2.5),

    env.new('mqf.1', xt.Quadrupole, length=0.6, k1='mq1'),
    env.new('d1.1',  xt.Drift, length=2.5),
    env.new('mk.1', xt.Marker),
    env.new('d1.2',  xt.Drift, length=2.5),

    env.new('mqd.1', xt.Quadrupole, length=0.6, k1='-mq1'),
    env.new('d3.1',  xt.Drift, length=5),

    env.new('mqf.2', xt.Quadrupole, length=0.6, k1='mq2'),
    env.new('d1.3',  xt.Drift, length=2.5),
    env.new('mk.2', xt.Marker),
    env.new('d1.4',  xt.Drift, length=2.5),

    env.new('mqd.2', xt.Quadrupole, length=0.6, k1='-mq2'),
    env.new('d3.2',  xt.Drift, length=2.5),
])

kin_energy_0 = 50e6 # 50 MeV
line.particle_ref = xt.Particles(energy0=kin_energy_0 + xt.PROTON_MASS_EV, # total energy
                                 mass0=xt.PROTON_MASS_EV)

tw = line.twiss4d()

betx_tar = tw['betx', 'mk.2']
bety_tar = tw['bety', 'mk.2']

env['mq1'] = 0.45
env['mq2'] = 0.55

tw1 = line.twiss4d()
print('Before matching:')
print(f"  mq1 = {env['mq1']}, mq2 = {env['mq2']}")
print(f"  betx = {tw1['betx', 'mk.2']}, bety = {tw1['bety', 'mk.2']}")
print(f"  Target: betx = {betx_tar}, bety = {bety_tar}")

opt = line.match(
    solve=False,
    method='4d',
    vary=xt.VaryList(['mq1', 'mq2']),
    targets=[xt.TargetSet(at='mk.2', betx=betx_tar, bety=bety_tar, tol=1e-6)],
    use_ad=True
)

opt.step(1)

print('After matching:')
tw2 = line.twiss4d()
print(f"  mq1 = {env['mq1']}, mq2 = {env['mq2']}")
print(f"  betx = {tw2['betx', 'mk.2']}, bety = {tw2['bety', 'mk.2']}")

###### Graph part
# mf = opt._err
# # mf.quad_sources_ord

import pydot
import re
import copy
import os

def get_and_remove_all_node_names_downwards(subgraph):
    """
    Recursively get and remove all node names in a subgraph and its subgraphs.
    This is done inplace, no new graph is created.

    Parameters:
    -----------
    subgraph: pydot.Subgraph
        The input pydot subgraph.

    Returns:
    --------
    set of str
        A set of all node names that were removed.
    """
    node_names = {n.get_name().strip('"') for n in subgraph.get_nodes()}
    for i in node_names:
        subgraph.del_node(i)
    for sg in subgraph.get_subgraphs():
        node_names.update(get_and_remove_all_node_names_downwards(sg))
    return node_names

def _modify_edges(graph, subgraph, removed_node_names, replacement_node_name):
    # Doesn't work recursively yet
    edge_list = subgraph.get_edges()
    for e in edge_list:
        pts = e.obj_dict['points']
        if pts[0] in removed_node_names:
            if pts[1] in removed_node_names:
                subgraph.del_edge(pts[0], pts[1])
            else:
                e.obj_dict['points'] = (replacement_node_name, pts[1])
        else:
            if pts[1] in removed_node_names:
                e.obj_dict['points'] = (pts[0], replacement_node_name)
    for sg in subgraph.get_subgraphs():
        _modify_edges(graph, sg, removed_node_names, replacement_node_name)

def safe_clone_pydot_graph(graph):
    """
    Create a safe clone of a pydot graph by writing it to a temporary file and
    reading it back, removing any spurious newline nodes.
    Parameters:
    -----------
    graph: pydot.Dot
        The input pydot graph to clone.

    Returns:
    --------
    pydot.Dot
        The cloned pydot graph.
    """

    fname = "temp.dot"
    graph.write_raw(fname)
    try:
        new_graph = pydot.graph_from_dot_file(fname)[0]
    finally:
        os.remove(fname)
    for node in new_graph.get_nodes():
        if node.get_name() in ["\n", '"\\n"']:
            new_graph.del_node(node)
    return new_graph

def safe_load_pydot_from_file(fname):
    """
    Load a pydot graph from a dot file, removing any spurious newline nodes.

    Parameters:
    -----------
    fname: str
        Path to the dot file.

    Returns:
    --------
    pydot.Dot
        The loaded pydot graph.
    """

    graph = pydot.graph_from_dot_file(fname)[0]
    for node in graph.get_nodes():
        if node.get_name() in ["\n", '"\\n"']:
            graph.del_node(node)
    return graph

def collapse_selected_clusters(dot_graph, patterns=("quad", "bend", "drift")):
    """
    Collapse subgraphs (clusters) whose names match any of the given patterns.
    The collapsed subgraph is replaced by a single node with the label of the subgraph

    Parameters:
    -----------
    dot_graph: pydot.Dot
        The input pydot graph to modify.
    patterns: tuple of str
        Patterns to match subgraph names. If a subgraph name contains any of these patterns, it will be collapsed.

    Returns:
    --------
    pydot.Dot
        The modified pydot graph with selected subgraphs collapsed.
    """
    pattern = re.compile("|".join(patterns), re.IGNORECASE)

    # make copy of dot_graph
    dot_graph = safe_clone_pydot_graph(dot_graph)

    def _collapse_recursive(graph, topgraph):
        # If subgraph should be removed: Remove all nodes inside, but copy edges
        # to nodes outside the subgraph to the new collapsed node
        removed_node_names = []

        for sg in graph.get_subgraphs():
            sg_name = sg.get_name().strip('"').replace(' ', '')
            if pattern.search(sg_name):
                # -> remove subgraphs, all nodes and edges that are not here
                # for every edge, look up the parent graph(s) too
                # collect node names that will be removed

                remove_node_names = get_and_remove_all_node_names_downwards(sg)

                # create a collapsed node
                label = sg.get_label() or sg_name
                label_trimmed = label.strip('"').replace(' ', '')
                collapsed_node = pydot.Node(f"{label_trimmed}")
                graph.add_node(collapsed_node)

                # Traverse entire graph top-down to delete edges
                _modify_edges(topgraph, topgraph, remove_node_names, f"{label_trimmed}")

                # Delete Subgraph
                del graph.obj_dict['subgraphs'][sg_name]
            else:
                # Recursively go through this
                _collapse_recursive(sg, topgraph)

        return removed_node_names

    # collapse clusters recursively
    _collapse_recursive(dot_graph, dot_graph)

    return dot_graph


def create_test_graph():
    graph = pydot.Dot("TestGraph", graph_type="digraph")

    # Node outside clusters
    node_A = pydot.Node("A")
    node_E = pydot.Node("E")
    graph.add_node(node_A)
    graph.add_node(node_E)

    # Inner "quad" cluster containing B, C, D
    cluster_quad = pydot.Cluster(
        graph_name="cluster_quad",
        label="quad",
        style="dashed",
        color="gray"
    )
    node_B = pydot.Node("B")
    node_C = pydot.Node("C")
    node_D = pydot.Node("D")
    for n in [node_B, node_C, node_D]:
        cluster_quad.add_node(n)

    # cluster_quad.add_edge(pydot.Edge("B", "C"))
    # cluster_quad.add_edge(pydot.Edge("B", "D"))

    # Outer "arbit" cluster containing "arbit_inner" node and the "quad" subgraph
    cluster_arbit = pydot.Cluster(
        graph_name="cluster_arbit",
        label="arbit",
        style="dashed",
        color="blue"
    )
    node_arbit_inner = pydot.Node("arbit_inner")
    cluster_arbit.add_node(node_arbit_inner)
    cluster_arbit.add_subgraph(cluster_quad)

    #cluster_arbit.add_edge(pydot.Edge("D", "arbit_inner"))

    graph.add_subgraph(cluster_arbit)

    # Define edges
    graph.add_edge(pydot.Edge("A", "B"))  # A → B (inside quad)
    graph.add_edge(pydot.Edge("B", "C"))
    graph.add_edge(pydot.Edge("B", "D"))
    graph.add_edge(pydot.Edge("D", "arbit_inner"))
    graph.add_edge(pydot.Edge("A", "E"))

    return graph


# Create and save
g = create_test_graph()
g.write_raw("test_graph.dot")
g.write_png("test_graph.png")

g2 = collapse_selected_clusters(g, patterns=("quad", "bend", "drift"))
g2.write_raw("test_graph2.dot")
g2.write_png("test_graph2.png")

graph = safe_load_pydot_from_file("graph_autodiff.dot")
graph.write_png("graph_autodiff2.png")

graph2 = collapse_selected_clusters(graph, patterns=("quad", "bend", "drift", "branch"))
graph2.write_raw("graph_autodiff_collapsed.dot")
graph2.write_png("graph_autodiff_collapsed.png")