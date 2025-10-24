# dag_tensorflow_qubits.py
"""
DAG module using TensorFlow where:
- Source nodes: 0 incoming, 1 outgoing; each source holds a 2-vector (pair of complex numbers)
  whose squared-modulus sum equals 1 (normalised).
- Sink nodes: 1 incoming, 0 outgoing. Number of sources == number of sinks.
- Inner nodes: k incoming and k outgoing. Each inner node has a complex square matrix of shape
  (2**k, 2**k). Incoming edge vectors are Kronecker-multiplied (in the order edges were added)
  into a column vector of length 2**k, multiplied by the matrix, giving an output vector of length
  2**k. That output vector is assigned to every outgoing edge of the inner node.

Edges carry 1-D tensors (tf.Tensor) of arbitrary length; the module uses tf.complex128 by default.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import tensorflow as tf


class GraphError(Exception):
    pass


class Edge:
    """Directed edge carrying a 1-D tf.Tensor (complex) as its current value."""

    def __init__(self, name: str, start: "Node", end: "Node", dtype=tf.complex128):
        self.name = name
        self.start = start
        self.end = end
        self.dtype = dtype
        # store a 1-D tf.Tensor (initially zero scalar vector)
        self.value: tf.Tensor = tf.convert_to_tensor([0], dtype=self.dtype)

    def set_value(self, v: Any):
        """Set the edge's tensor value (accepts python/numpy/tf types)."""
        t = tf.convert_to_tensor(v, dtype=self.dtype)
        # Ensure 1-D vector
        if tf.rank(t) == 0:
            t = tf.reshape(t, [1])
        else:
            t = tf.reshape(t, [-1])
        self.value = tf.identity(t)

    def get_value(self) -> tf.Tensor:
        return tf.identity(self.value)

    def __repr__(self) -> str:
        return f"Edge({self.name}: {self.start.name}->{self.end.name}, shape={tuple(self.value.shape)})"


class Node:
    def __init__(self, name: str):
        self.name = name
        self.in_edges: List[Edge] = []
        self.out_edges: List[Edge] = []

    def __repr__(self):
        return f"Node({self.name})"


class SourceNode(Node):
    def __init__(self, name: str, dtype=tf.complex128):
        super().__init__(name)
        self.dtype = dtype
        # store a length-2 tf.Tensor representing the pair of complex values (normalized)
        self._value: tf.Tensor = tf.convert_to_tensor([0, 0], dtype=self.dtype)

    def set_value(self, pair: Any):
        t = tf.convert_to_tensor(pair, dtype=self.dtype)
        t = tf.reshape(t, [-1])
        if t.shape[0] != 2:
            raise GraphError(f"Source {self.name} initializer must be a pair (length 2), got shape {t.shape}")
        # check normalization: sum of squared moduli equals 1 (within tolerance)
        mod_sq = tf.reduce_sum(tf.math.real(tf.math.conj(t) * t))
        # allow small numeric tolerance
        if not tf.math.abs(mod_sq - tf.constant(1.0, dtype=mod_sq.dtype)) < tf.constant(1e-8, dtype=mod_sq.dtype):
            raise GraphError(f"Source {self.name} initializer must be normalized: sum|v|^2 == 1 (got {mod_sq.numpy()})")
        self._value = tf.identity(t)

    def get_value(self) -> tf.Tensor:
        return tf.identity(self._value)

    def __repr__(self):
        return f"SourceNode({self.name}, value_shape={tuple(self._value.shape)})"


class SinkNode(Node):
    def __init__(self, name: str):
        super().__init__(name)


class InnerNode(Node):
    def __init__(self, name: str, degree: int, matrix: Optional[Any] = None, dtype=tf.complex128):
        """
        degree = number of incoming (and outgoing) edges = k
        matrix: optional (2**k, 2**k) complex matrix; if omitted, random complex init used.
        """
        super().__init__(name)
        if degree <= 0:
            raise ValueError("degree must be positive")
        self.degree = degree
        self.dtype = dtype
        size = 2 ** degree
        if matrix is not None:
            m = tf.convert_to_tensor(matrix, dtype=self.dtype)
            m = tf.reshape(m, (size, size))
            self.matrix = tf.identity(m)
        else:
            # random complex init: small gaussian real+imag
            float_dtype = tf.float64 if self.dtype == tf.complex128 else tf.float32
            real = tf.random.normal((size, size), stddev=0.1, dtype=float_dtype)
            imag = tf.random.normal((size, size), stddev=0.1, dtype=float_dtype)
            comp = tf.complex(real, imag)
            self.matrix = tf.identity(tf.cast(comp, dtype=self.dtype))

    def set_matrix(self, mat: Any):
        size = 2 ** self.degree
        m = tf.convert_to_tensor(mat, dtype=self.dtype)
        m = tf.reshape(m, (size, size))
        self.matrix = tf.identity(m)

    def get_matrix(self) -> tf.Tensor:
        return tf.identity(self.matrix)

    def __repr__(self):
        return f"InnerNode({self.name}, degree={self.degree}, matrix_shape={tuple(self.matrix.shape)})"


class Graph:
    def __init__(self, dtype=tf.complex128):
        self.dtype = dtype
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._edge_counter = 0

    def add_source(self, name: str) -> SourceNode:
        if name in self.nodes:
            raise GraphError(f"Node {name} already exists")
        s = SourceNode(name, dtype=self.dtype)
        self.nodes[name] = s
        return s

    def add_sink(self, name: str) -> SinkNode:
        if name in self.nodes:
            raise GraphError(f"Node {name} already exists")
        s = SinkNode(name)
        self.nodes[name] = s
        return s

    def add_inner(self, name: str, degree: int, matrix: Optional[Any] = None) -> InnerNode:
        if name in self.nodes:
            raise GraphError(f"Node {name} already exists")
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

    def initialize_sources(self, initializer: Dict[str, Any]):
        """
        initializer: mapping source_name -> length-2 iterable (complex)
        Each vector must be normalized (sum |v|^2 == 1).
        """
        for name, val in initializer.items():
            if name not in self.nodes:
                raise GraphError(f"Source {name} not found")
            node = self.nodes[name]
            if not isinstance(node, SourceNode):
                raise GraphError(f"Node {name} is not a SourceNode")
            node.set_value(val)

    def set_inner_matrix(self, node_name: str, matrix: Any):
        n = self.nodes.get(node_name)
        if not isinstance(n, InnerNode):
            raise GraphError("set_inner_matrix applies only to InnerNode")
        n.set_matrix(matrix)

    def validate(self):
        # structural checks
        source_count = 0
        sink_count = 0
        for name, node in self.nodes.items():
            if isinstance(node, SourceNode):
                source_count += 1
                if len(node.in_edges) != 0:
                    raise GraphError(f"Source node {name} must have 0 incoming edges")
                if len(node.out_edges) != 1:
                    raise GraphError(f"Source node {name} must have exactly 1 outgoing edge (has {len(node.out_edges)})")
            elif isinstance(node, SinkNode):
                sink_count += 1
                if len(node.out_edges) != 0:
                    raise GraphError(f"Sink node {name} must have 0 outgoing edges")
                if len(node.in_edges) != 1:
                    raise GraphError(f"Sink node {name} must have exactly 1 incoming edge (has {len(node.in_edges)})")
            elif isinstance(node, InnerNode):
                if len(node.in_edges) != node.degree:
                    raise GraphError(f"Inner node {name} expected {node.degree} incoming edges but found {len(node.in_edges)}")
                if len(node.out_edges) != node.degree:
                    raise GraphError(f"Inner node {name} expected {node.degree} outgoing edges but found {len(node.out_edges)}")
                # matrix shape
                expected = 2 ** node.degree
                mat = node.get_matrix()
                if tuple(mat.shape) != (expected, expected):
                    raise GraphError(f"Inner node {name} matrix shape must be ({expected},{expected})")
            else:
                raise GraphError(f"Unknown node type for {name}")

        if source_count != sink_count:
            raise GraphError(f"Number of sources ({source_count}) must equal number of sinks ({sink_count})")

        # acyclicity check (Kahn)
        if not self._is_acyclic():
            raise GraphError("Graph must be acyclic")

    def _is_acyclic(self) -> bool:
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

    @staticmethod
    def _kron_list(vecs: List[tf.Tensor]) -> tf.Tensor:
        """Compute Kronecker (tensor) product of a list of 1-D tensors in order.
        Result is a 1-D tensor of length prod(len(v) for v in vecs)."""
        if not vecs:
            # by convention, kron of empty list is scalar 1
            return tf.convert_to_tensor([1], dtype=vecs[0].dtype if vecs else tf.complex128)
        res = tf.reshape(vecs[0], [-1])
        for v in vecs[1:]:
            a = tf.reshape(res, [-1])
            b = tf.reshape(v, [-1])
            # tensordot with axes=0 then flatten
            t = tf.tensordot(a, b, axes=0)
            res = tf.reshape(t, [-1])
        return tf.cast(res, dtype=vecs[0].dtype)

    def simulate(self):
        """
        One-step propagation:
        - Validate graph.
        - For each source, set its single outgoing edge to the source's 2-vector.
        - For each inner node:
            * collect incoming edge vectors (in the order they appear in node.in_edges)
            * compute kronecker product -> column vector length 2**k
            * compute output = matrix @ input_vector
            * assign output vector (1-D) to every outgoing edge of that inner node
        """
        self.validate()

        # 1) propagate sources
        for node in self.nodes.values():
            if isinstance(node, SourceNode):
                if node.out_edges:
                    val = node.get_value()  # length-2
                    node.out_edges[0].set_value(val)

        # 2) propagate inner nodes
        for node in self.nodes.values():
            if isinstance(node, InnerNode):
                # gather incoming vectors in order
                in_vecs = []
                for e in node.in_edges:
                    v = e.get_value()
                    # ensure 1-D
                    v = tf.reshape(v, [-1])
                    in_vecs.append(tf.cast(v, dtype=self.dtype))
                # build kronecker product
                joint = self._kron_list(in_vecs)  # 1-D length 2**k
                joint_col = tf.reshape(joint, [-1, 1])  # column vector
                # matmul
                mat = node.get_matrix()
                out_col = tf.matmul(mat, joint_col)  # shape (2**k, 1)
                out_vec = tf.reshape(out_col, [-1])  # 1-D
                # assign to every outgoing edge
                for e in node.out_edges:
                    e.set_value(out_vec)

    def get_edge_values(self) -> Dict[str, tf.Tensor]:
        """Return mapping edge_name -> tf.Tensor (copy)."""
        return {e.name: tf.identity(e.get_value()) for e in self.edges}

    def get_edge_values_numpy(self) -> Dict[str, Any]:
        """Return mapping edge_name -> numpy array / python complex array (for convenience)."""
        out = {}
        for e in self.edges:
            try:
                out[e.name] = e.get_value().numpy()
            except Exception:
                out[e.name] = None
        return out

    def get_node_summary(self) -> Dict[str, Dict[str, Any]]:
        d = {}
        for name, node in self.nodes.items():
            d[name] = {
                "type": type(node).__name__,
                "in": [e.name for e in node.in_edges],
                "out": [e.name for e in node.out_edges]
            }
            if isinstance(node, InnerNode):
                d[name]["degree"] = node.degree
                d[name]["matrix_shape"] = tuple(node.get_matrix().shape)
        return d


if __name__ == "__main__":
    # Example usage / small self-test
    # - Two sources s0 and s1 (each a 2-vector, normalized)
    # - One inner node A with degree 2 (matrix 4x4)
    # - One sink t (one incoming)
    #
    # Graph layout:
    #   s0 --> A --
    #             \\--> t (A has 2 outgoing edges; both go to t)
    #   s1 --> A --
    #
    # Number sources == number sinks? We'll add one sink per source: here 2 sources -> 2 sinks;
    # for a short example we connect both A outputs to the same sink name twice to satisfy
    # "each sink must have exactly 1 incoming edge" constraint; in real usage sinks would be
    # distinct nodes.
    g = Graph(dtype=tf.complex128)

    # add sources/sinks
    s0 = g.add_source("s0")
    s1 = g.add_source("s1")
    t0 = g.add_sink("t0")
    t1 = g.add_sink("t1")

    # inner node degree 2 (2 incoming and 2 outgoing)
    A = g.add_inner("A", degree=2)

    # edges
    g.add_edge("s0", "A")  # s0 -> A (incoming 0)
    g.add_edge("s1", "A")  # s1 -> A (incoming 1)
    g.add_edge("A", "t0")  # A outgoing 0 -> t0
    g.add_edge("A", "t1")  # A outgoing 1 -> t1

    # set matrix for A: identity 4x4
    g.set_inner_matrix("A", tf.eye(4, dtype=tf.complex128))

    # init sources: normalized 2-vectors
    g.initialize_sources({
        "s0": [1.0 + 0.0j, 0.0 + 0.0j],  # |0>
        "s1": [0.0 + 0.0j, 1.0 + 0.0j],  # |1>
    })

    # run simulation
    g.simulate()

    # print edge values as numpy arrays
    ev = g.get_edge_values_numpy()
    print("Edge values (numpy):")
    for k, v in ev.items():
        print(k, v)

    # node summary
    print("\nNode summary:")
    import pprint
    pprint.pprint(g.get_node_summary())
