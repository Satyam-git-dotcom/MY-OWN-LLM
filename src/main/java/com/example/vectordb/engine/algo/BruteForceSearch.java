package com.example.vectordb.engine.algo;

import com.example.vectordb.engine.db.VectorItem;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;

public class BruteForceSearch {
    private final List<VectorItem> items = new ArrayList<>();

    public synchronized void insert(VectorItem item) {
        items.add(item);
    }

    public synchronized void remove(int id) {
        items.removeIf(item -> item.getId() == id);
    }

    public synchronized List<SearchHit> knn(List<Float> q, int k, BiFunction<List<Float>, List<Float>, Float> dist) {
        List<SearchHit> results = new ArrayList<>(items.size());
        for (VectorItem item : items) {
            float d = dist.apply(q, item.getEmb());
            results.add(new SearchHit(item.getId(), d));
        }
        Collections.sort(results);
        if (results.size() > k) {
            return new ArrayList<>(results.subList(0, k));
        }
        return results;
    }

    public synchronized List<VectorItem> getItems() {
        return new ArrayList<>(items);
    }
}
