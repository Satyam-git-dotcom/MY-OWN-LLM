package com.example.vectordb.engine.algo;

import com.example.vectordb.engine.db.VectorItem;
import java.util.*;
import java.util.function.BiFunction;

public class KDTree {
    
    private static class KDNode {
        VectorItem item;
        KDNode left;
        KDNode right;

        KDNode(VectorItem item) {
            this.item = item;
        }
    }

    private KDNode root;
    private final int dims;

    public KDTree(int dims) {
        this.dims = dims;
    }

    public synchronized void insert(VectorItem item) {
        root = insertRec(root, item, 0);
    }

    private KDNode insertRec(KDNode node, VectorItem item, int depth) {
        if (node == null) {
            return new KDNode(item);
        }
        int axis = depth % dims;
        if (item.getEmb().get(axis) < node.item.getEmb().get(axis)) {
            node.left = insertRec(node.left, item, depth + 1);
        } else {
            node.right = insertRec(node.right, item, depth + 1);
        }
        return node;
    }

    public synchronized void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem item : items) {
            insert(item);
        }
    }

    public synchronized List<SearchHit> knn(List<Float> q, int k, BiFunction<List<Float>, List<Float>, Float> dist) {
        if (root == null || k <= 0) {
            return Collections.emptyList();
        }
        // Max-heap: largest distance at the top so we can pop it when size > k
        PriorityQueue<SearchHit> heap = new PriorityQueue<>((a, b) -> Float.compare(b.distance(), a.distance()));
        knnRec(root, q, k, 0, dist, heap);
        
        List<SearchHit> results = new ArrayList<>(heap);
        Collections.sort(results);
        return results;
    }

    private void knnRec(KDNode node, List<Float> q, int k, int depth, 
                        BiFunction<List<Float>, List<Float>, Float> dist, 
                        PriorityQueue<SearchHit> heap) {
        if (node == null) {
            return;
        }
        
        float dn = dist.apply(q, node.item.getEmb());
        if (heap.size() < k || dn < heap.peek().distance()) {
            heap.offer(new SearchHit(node.item.getId(), dn));
            if (heap.size() > k) {
                heap.poll();
            }
        }
        
        int axis = depth % dims;
        float diff = q.get(axis) - node.item.getEmb().get(axis);
        KDNode closer = diff < 0 ? node.left : node.right;
        KDNode farther = diff < 0 ? node.right : node.left;
        
        knnRec(closer, q, k, depth + 1, dist, heap);
        
        if (heap.size() < k || Math.abs(diff) < heap.peek().distance()) {
            knnRec(farther, q, k, depth + 1, dist, heap);
        }
    }
}
