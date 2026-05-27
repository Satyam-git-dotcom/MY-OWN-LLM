package com.example.vectordb.engine.db;

import com.example.vectordb.engine.algo.*;
import java.util.*;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.BiFunction;

public class VectorDatabase {

    public record SearchHitDetail(int id, String metadata, String category, List<Float> emb, float distance) {}
    public record SearchOut(List<SearchHitDetail> results, long latencyUs, String algo, String metric, List<TraversalStep> traversalPath) {}
    public record BenchOut(long bruteforceUs, long kdtreeUs, long hnswUs, int itemCount) {}

    private final Map<Integer, VectorItem> store = new HashMap<>();
    private final BruteForceSearch bf = new BruteForceSearch();
    private final KDTree kdt;
    private final HNSW hnsw;
    private final ReentrantReadWriteLock rwLock = new ReentrantReadWriteLock();
    private int nextId = 1;
    public final int dims;

    public VectorDatabase(int dims) {
        this.dims = dims;
        this.kdt = new KDTree(dims);
        this.hnsw = new HNSW(16, 200);
    }

    public int insert(String meta, String cat, List<Float> emb, BiFunction<List<Float>, List<Float>, Float> dist) {
        rwLock.writeLock().lock();
        try {
            int id = nextId++;
            VectorItem v = new VectorItem(id, meta, cat, emb);
            store.put(id, v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            return id;
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    public boolean remove(int id) {
        rwLock.writeLock().lock();
        try {
            if (!store.containsKey(id)) {
                return false;
            }
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            kdt.rebuild(new ArrayList<>(store.values()));
            return true;
        } finally {
            rwLock.writeLock().unlock();
        }
    }

    public SearchOut search(List<Float> q, int k, String metric, String algo) {
        rwLock.readLock().lock();
        try {
            BiFunction<List<Float>, List<Float>, Float> dist = DistanceMetrics.getDistanceFn(metric);
            long t0 = System.nanoTime();

            List<SearchHit> raw;
            List<TraversalStep> traversalPath = Collections.emptyList();

            if ("bruteforce".equalsIgnoreCase(algo)) {
                raw = bf.knn(q, k, dist);
            } else if ("kdtree".equalsIgnoreCase(algo)) {
                raw = kdt.knn(q, k, dist);
            } else {
                HNSWKnnResult hnswResult = hnsw.knn(q, k, 50, dist);
                raw = hnswResult.hits();
                traversalPath = hnswResult.traversalPath();
            }

            long latencyUs = (System.nanoTime() - t0) / 1000;

            List<SearchHitDetail> results = new ArrayList<>();
            for (SearchHit hit : raw) {
                VectorItem item = store.get(hit.id());
                if (item != null) {
                    results.add(new SearchHitDetail(hit.id(), item.getMetadata(), item.getCategory(), item.getEmb(), hit.distance()));
                }
            }

            return new SearchOut(results, latencyUs, algo, metric, traversalPath);
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public BenchOut benchmark(List<Float> q, int k, String metric) {
        rwLock.readLock().lock();
        try {
            BiFunction<List<Float>, List<Float>, Float> dist = DistanceMetrics.getDistanceFn(metric);
            
            long tBf0 = System.nanoTime();
            bf.knn(q, k, dist);
            long tBf = (System.nanoTime() - tBf0) / 1000;

            long tKd0 = System.nanoTime();
            kdt.knn(q, k, dist);
            long tKd = (System.nanoTime() - tKd0) / 1000;

            long tHnsw0 = System.nanoTime();
            hnsw.knn(q, k, 50, dist);
            long tHnsw = (System.nanoTime() - tHnsw0) / 1000;

            return new BenchOut(tBf, tKd, tHnsw, store.size());
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public List<VectorItem> all() {
        rwLock.readLock().lock();
        try {
            return new ArrayList<>(store.values());
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public HNSW.GraphInfo hnswInfo() {
        rwLock.readLock().lock();
        try {
            return hnsw.getInfo();
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
}
