"""Function for detecting communities based on Louvain Community Detection
Algorithm"""

import itertools
from collections import defaultdict, deque

import networkx as nx
import copy
from networkx.utils import py_random_state
from networkx.algorithms.community.louvain import _neighbor_weights

def balancedness():
    return
