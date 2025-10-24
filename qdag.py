"""
dag_tensorflow.py

A small Python module that implements a directed acyclic graph (DAG) where:
- Source nodes: 1 outgoing edge, 0 incoming edges. Each source has an initializer value.
- Sink nodes: 1 incoming edge, 0 outgoing edges.
- Inner nodes: same number of incoming and outgoing edges (call it n). Each inner node
  has a square complex matrix of shape (n, n). The multiplication mapping incoming
  column vector -> outgoing column vector is `out = matrix @ in`.

This implementation uses TensorFlow tensors/variables and matrix operations. It
is implemented to be easy to use in eager mode and also supports decorating
`simulate` with `tf.function` for speed.

Example (short):
    from dag_tensorflow import Graph, SourceNode, InnerNode, SinkNode
    import tensorflow as tf

    g = Graph(dtype=tf.complex128)
    s1 = g.add_source('s1')
    s2 = g.add_source('s2')
    a = g.add_inner('A', degree=2)
    t = g.add_sink('t')

    # build edges (ordering matters for matrix rows/cols)
    g.add_edge('s1', 'A')   # this will be one of A's incoming edges
    g.add_edge('s2', 'A')   # second incoming
    g.add_edge('A', 't')    # A must have 2 outgoing edges; to satisfy degree=2
    g.add_edge('A', 't')    # (you may connect to the same sink multiple times if graph requires)

    # set A's matrix explicitly or rely on random initialization
    g.set_inner_matrix('A', tf.eye(2, dtype=tf.complex128))

    # initialize sources
    g.initialize_sources({'s1': 1+0j, 's2': 2+0j})

    # run simulation: this propagates source values along edges and applies inner matrices
    g.simulate()

    # inspect edge values
    print(g.get_edge_values())

Note: The graph is validated at `graph.validate()`; it checks the structural
constraints described above and raises informative errors if the constraints are
violated.

"""

from __future__ import annotations

import tensorflow as tf
from typing import Any, Dict, List, Optional, Sequence


class GraphError(Exception):
    pass


class Edge:
    """Directed edge carrying a complex scalar tensor value."""

    def __init__(self, name: str, start: "Node", end: "Node", dtype=tf.complex128):
        self.name = name
        self.start = start
        self.end = end
        # scalar complex variable holding the current value on this edge
        self.dtype = dtype
        self.value = tf.Variable(tf.cast(0, dtype=self.dtype), trainable=False)

    def set_value(self, v: Any):
        t = tf.convert_to_tensor(v, dtype=self.dtype)
        # allow scalar shapes only
        t = tf.reshape(t, [])
        self.value.assign(t)

    def get_value(self) -> tf.Tensor:
        return tf.identity(self.value)

    def __repr__(self) -> str:
        return f"Edge({self.name}: {self.start.name}->{self.end.name}, value={self.value.numpy()})"


class Node:
    def __init__(self, name: str):
        self.name = name
        self.in_edges: List[Edge] = []
        self.out_edges: List[Edge] = []

    def __repr__(self) -> str:
        return f"Node({self.name})"


class SourceNode(Node):
    def __init__(self, name: str, dtype=tf.complex128):
        super().__init__(name)
        self.dtype = dtype
        # scalar value assigned by initializer
        self._value = tf.Variable(tf.cast(0, dtype=self.dtype), trainable=False)

    def set_value(self, v: Any):
        t = tf.convert_to_tensor(v, dtype=self.dtype)
        t = tf.reshape(t, [])
        self._value.assign(t)

    def get_value(self) -> tf.Tensor:
        return tf.identity(self._value)

    def __repr__(self) -> str:
        return f"SourceNode({self.name}, value={self._value.numpy()})"


class SinkNode(Node):
    def __init__(self, name: str):
        super().__init__(name)


class InnerNode(Node):
    def __init__(self, name: str, degree: int, matrix: Optional[tf.Tensor] = None, dtype=tf.complex128):
        """
        Inner node with `degree` incoming and `degree` outgoing edges.

        Parameters
        ----------
        name: identifier
        degree: number of incoming/outgoing edges
        matrix: optional initial square matrix (degree x degree). If omitted,
                the matrix is initialized randomly.
        dtype: tensorflow dtype (tf.complex64 or tf.complex128 recommended)
        """
        super().__init__(name)
        if degree <= 0:
            raise ValueError("degree must be positive")
        self.degree = degree
        self.dtype = dtype
        if matrix is not None:
            mat = tf.convert_to_tensor(matrix, dtype=self.dtype)
            mat = tf.reshape(mat, (degree, degree))
            self.matrix = tf.Variable(mat, trainable=False)
        else:
            # random complex initialization: real and imag normal(0, 0.1)
            real = tf.random.normal((degree, degree), stddev=0.1, dtype=tf.float64 if self.dtype==tf.complex128 else tf.float32)
            imag = tf.random.normal((degree, degree), stddev=0.1, dtype=tf.float64 if self.dtype==tf.complex128 else tf.float32)
            comp = tf.complex(real, imag)
            self.matrix = tf.Variable(tf.cast(comp, dtype=self.dtype), trainable=False)

    def set_matrix(self, mat: Any):
        m = tf.convert_to_tensor(mat, dtype=self.dtype)
        m = tf.reshape(m, (self.degree, self.degree))
        self.matrix.assign(m)

    def get_matrix(self) -> tf.Tensor:
        return tf.identity(self.matrix)

    def __repr__(self) -> str:
        return f"InnerNode({self.name}, degree={self.degree})"


class Graph:
    def __init__(self, dtype=tf.complex128):
        self.dtype = dtype
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._edge_counter = 0

    # node creation helpers
    def add_source(self, name: str) -> SourceNode:
        if name in self.nodes:
            raise GraphError(f"Node with name {name} already exists")
        s = SourceNode(name, dtype=self.dtype)
        self.nodes[name] = s
        return s

    def add_sink(self, name: str) -> SinkNode:
        if name in self.nodes:
            raise GraphError(f"Node with name {name} already exists")
        s = SinkNode(name)
        self.nodes[name] = s
        return s

    def add_inner(self, name: str, degree: int, matrix: Optional[tf.Tensor] = None) -> InnerNode:
        if name in self.nodes:
            raise GraphError(f"Node with name {name} already exists")
        n = InnerNode(name, degree=degree, matrix=matrix, dtype=self.dtype)
        self.nodes[name] = n
        return n

    def add_edge(self, start_name: str, end_name: str) -> Edge:
        if start_name not in self.nodes:
            raise GraphError(f"Start node {start_name} not found")
        if end_name not in self.nodes:
            raise GraphError(f"End node {end_name} not found")
        start = self.nodes[start_name]
        end = self.nodes[end_name]
        eid = self._edge_counter
        self._edge_counter += 1
        edge = Edge(f"e{eid}", start=start, end=end, dtype=self.dtype)
        start.out_edges.append(edge)
        end.in_edges.append(edge)
        self.edges.append(edge)
        return edge

    def set_inner_matrix(self, node_name: str, matrix: Any):
        n = self.nodes.get(node_name)
        if not isinstance(n, InnerNode):
            raise GraphError("set_inner_matrix applies only to InnerNode")
        n.set_matrix(matrix)

    def initialize_sources(self, initializer: Dict[str, Any]):
        """
        initializer: mapping from source node name -> scalar (Python complex, numeric, or tf tensor)
        """
        for name, val in initializer.items():
            if name not in self.nodes:
                raise GraphError(f"Source {name} not found")
            node = self.nodes[name]
            if not isinstance(node, SourceNode):
                raise GraphError(f"Node {name} is not a SourceNode")
            node.set_value(val)

    def validate(self):
        """Validate graph constraints described in the module docstring."""
        # Check node counts and degrees
        for name, node in self.nodes.items():
            if isinstance(node, SourceNode):
                if len(node.in_edges) != 0:
                    raise GraphError(f"Source node {name} must have 0 incoming edges")
                if len(node.out_edges) != 1:
                    raise GraphError(f"Source node {name} must have exactly 1 outgoing edge (has {len(node.out_edges)})")
            elif isinstance(node, SinkNode):
                if len(node.out_edges) != 0:
                    raise GraphError(f"Sink node {name} must have 0 outgoing edges")
                if len(node.in_edges) != 1:
                    raise GraphError(f"Sink node {name} must have exactly 1 incoming edge (has {len(node.in_edges)})")
            elif isinstance(node, InnerNode):
                if len(node.in_edges) != node.degree:
                    raise GraphError(f"Inner node {name} expected {node.degree} incoming edges but found {len(node.in_edges)}")
                if len(node.out_edges) != node.degree:
                    raise GraphError(f"Inner node {name} expected {node.degree} outgoing edges but found {len(node.out_edges)}")
                # matrix shapes
                mat = node.get_matrix()
                if mat.shape != (node.degree, node.degree):
                    raise GraphError(f"Inner node {name} matrix shape must be ({node.degree},{node.degree})")
            else:
                raise GraphError(f"Unknown node type for {name}")

        # Optional: check acyclicity by topological sort
        if not self._is_acyclic():
            raise GraphError("Graph must be acyclic")

    def _is_acyclic(self) -> bool:
        # Kahn's algorithm on node names
        indeg = {name: len(n.in_edges) for name, n in self.nodes.items()}
        Q = [name for name, d in indeg.items() if d == 0]
        visited = 0
        while Q:
            nname = Q.pop()
            visited += 1
            node = self.nodes[nname]
            for e in node.out_edges:
                indeg[e.end.name] -= 1
                if indeg[e.end.name] == 0:
                    Q.append(e.end.name)
        return visited == len(self.nodes)

    @tf.function
    def _propagate_inner(self, matrices: Sequence[tf.Tensor], in_vectors: Sequence[tf.Tensor], out_vars: Sequence[tf.Variable]):
        """
        Helper wrapped as tf.function to do batched matmul style propagation for inner nodes.
        Not strictly necessary but helps performance on larger graphs. The sequences should
        align in order.
        """
        for mat, inv, out_vs in zip(matrices, in_vectors, out_vars):
            # inv is shape [n, 1]; mat is [n, n]
            res = tf.matmul(mat, inv)  # shape [n, 1]
            # assign each element to corresponding outgoing edge variable
            # flatten to shape [n]
            flat = tf.reshape(res, [-1])
            for i in tf.range(tf.shape(flat)[0]):
                out_vs[i].assign(flat[i])

    def simulate(self):
        """
        Perform one step propagation:
        - For each source node, set its outgoing edge value to the source initializer value
        - For each inner node, gather incoming edge scalar values into a column vector (shape [n,1]),
          compute matrix @ vector and set outgoing edges accordingly.

        Graph must be validated prior to simulate() (call validate()).
        """
        # Validate first
        self.validate()

        # 1) propagate sources directly to their single outgoing edge
        for node in self.nodes.values():
            if isinstance(node, SourceNode):
                if node.out_edges:
                    val = node.get_value()
                    # set the single outgoing edge
                    node.out_edges[0].set_value(val)

        # 2) propagate inner nodes
        matrices = []
        in_vectors = []
        out_vars = []
        for node in self.nodes.values():
            if isinstance(node, InnerNode):
                # gather incoming edge scalar tensors in the order edges were added
                in_vals = [tf.reshape(e.get_value(), [1, 1]) for e in node.in_edges]
                # build column vector shape [n, 1]
                inv = tf.concat(in_vals, axis=0)
                matrices.append(node.get_matrix())
                in_vectors.append(inv)
                # collect outgoing edge variable references in same ordering as out_edges
                out_vars.append([e.value for e in node.out_edges])

        # use tf.function helper to do matmul and assign
        # However _propagate_inner expects python lists of tensors/vars, they will be traced
        # We call without tf.function here so it works in eager, but helper is tf.function
        # and will be used for the inner loop:
        if matrices:
            # call python-level helper that in turn calls tf.function per node
            self._propagate_inner(matrices, in_vectors, out_vars)

    def get_edge_values(self) -> Dict[str, complex]:
        """Return a mapping edge_name -> python complex value (numpy scalar)."""
        out = {}
        for e in self.edges:
            # convert tensor scalar to python complex
            v = e.get_value().numpy().item()
            out[e.name] = v
        return out

    def get_node_summary(self) -> Dict[str, Dict[str, Any]]:
        d = {}
        for name, node in self.nodes.items():
            d[name] = {
                'type': type(node).__name__,
                'in': [e.name for e in node.in_edges],
                'out': [e.name for e in node.out_edges]
            }
            if isinstance(node, InnerNode):
                d[name]['degree'] = node.degree
        return d


# If run as script, demonstrate a small example
if __name__ == '__main__':
    # small self-test demonstration
    g = Graph(dtype=tf.complex128)
    s1 = g.add_source('s1')
    s2 = g.add_source('s2')
    A = g.add_inner('A', degree=2)
    T = g.add_sink('t')

    g.add_edge('s1', 'A')
    g.add_edge('s2', 'A')
    g.add_edge('A', 't')
    g.add_edge('A', 't')

    # set inner matrix explicitly
    g.set_inner_matrix('A', tf.constant([[1+0j, 0+0j], [0+0j, 2+0j]], dtype=tf.complex128))
    g.initialize_sources({'s1': 1+0j, 's2': 3+0j})
    g.simulate()
    print('Edges:', g.get_edge_values())
    print('Nodes:', g.get_node_summary())
