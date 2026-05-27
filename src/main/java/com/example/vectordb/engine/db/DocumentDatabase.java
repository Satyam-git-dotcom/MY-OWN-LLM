package com.example.vectordb.engine.db;

import com.example.vectordb.engine.algo.BruteForceSearch;
import com.example.vectordb.engine.algo.DistanceMetrics;
import com.example.vectordb.engine.algo.HNSW;
import com.example.vectordb.engine.algo.SearchHit;
import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class DocumentDatabase {

    private final Map<Integer, DocItem> store = new HashMap<>();
    private final HNSW hnsw = new HNSW(16, 200);
    private final BruteForceSearch bf = new BruteForceSearch();
    private final ReentrantReadWriteLock rwLock = new ReentrantReadWriteLock();
    private int nextId = 1;
    private int dims = 0;

    public int insert(String title, String text, List<Float> emb) {
        rwLock.writeLock().lock();
        try {
            if (dims == 0) {
                dims = emb.size();
            }
            int id = nextId++;
            DocItem item = new DocItem(id, title, text, emb);
            store.put(id, item);

            // Replicate VectorItem interface for HNSW and BruteForce indexing
            VectorItem vi = new VectorItem(id, title, "doc", emb);
            hnsw.insert(vi, DistanceMetrics::cosine);
            bf.insert(vi);

            return id;
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    public List<DocSearchHit> search(List<Float> q, int k, float maxDist) {
        rwLock.readLock().lock();
        try {
            if (store.isEmpty()) {
                return Collections.emptyList();
            }
            List<SearchHit> raw;
            if (store.size() < 10) {
                raw = bf.knn(q, k, DistanceMetrics::cosine);
            } else {
                raw = hnsw.knn(q, k, 50, DistanceMetrics::cosine).hits();
            }

            List<DocSearchHit> out = new ArrayList<>();
            for (SearchHit hit : raw) {
                DocItem item = store.get(hit.id());
                if (item != null && hit.distance() <= maxDist) {
                    out.add(new DocSearchHit(hit.distance(), item));
                }
            }
            return out;
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public boolean remove(int id) {
        rwLock.writeLock().lock();
        try {
            if (!store.containsKey(id)) {
                return false;
            }
            store.remove(id);
            hnsw.remove(id);
            bf.remove(id);
            return true;
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    public List<DocItem> all() {
        rwLock.readLock().lock();
        try {
            return new ArrayList<>(store.values());
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public int size() {
        rwLock.readLock().lock();
        try {
            return store.size();
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public int getDims() {
        rwLock.readLock().lock();
        try {
            return dims;
        } finally {
            rwLock.readLock().unlock();
        }
    }
}
