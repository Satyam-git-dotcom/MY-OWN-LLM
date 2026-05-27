package com.example.vectordb.engine.algo;

public record SearchHit(int id, float distance) implements Comparable<SearchHit> {
    @Override
    public int compareTo(SearchHit other) {
        return Float.compare(this.distance, other.distance);
    }
}
