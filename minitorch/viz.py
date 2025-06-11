from graphviz import Digraph
import numpy as np

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def format_val(val):
    if isinstance(val, np.ndarray):
        if val.size == 1:
            return f"{val.item():.4f}"
        return np.array2string(val, precision=2, separator=',', suppress_small=True)
    try:
        return f"{val:.4f}"
    except:
        return str(val)

def draw_dot(root, format='svg', rankdir='LR'):
    """
    Visualize the computation graph for a Tensor using Graphviz.
    Supports numpy array data/grad.
    """
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
    
    for n in nodes:
        label = f"{{ data {format_val(n.data)} | grad {format_val(n.grad)} }}"
        dot.node(name=str(id(n)), label=label, shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot