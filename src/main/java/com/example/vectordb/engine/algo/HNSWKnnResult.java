package com.example.vectordb.engine.algo;

import java.util.List;

public record HNSWKnnResult(List<SearchHit> hits, List<TraversalStep> traversalPath) {}
