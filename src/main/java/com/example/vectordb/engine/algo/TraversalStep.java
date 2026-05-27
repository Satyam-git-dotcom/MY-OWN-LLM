package com.example.vectordb.engine.algo;

public record TraversalStep(int fromId, int toId, int layer, float distance) {}
