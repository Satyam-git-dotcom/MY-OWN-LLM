package com.example.vectordb.engine.algo;

import com.example.vectordb.engine.db.VectorItem;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;

public class HNSW {

    public static class Node {
        public VectorItem item;
        public int maxLyr;
        public List<List<Integer>> nbrs; // Layer -> List of neighbor node IDs

        public Node(VectorItem item, int maxLyr) {
            this.item = item;
            this.maxLyr = maxLyr;
            this.nbrs = new ArrayList<>();
            for (int i = 0; i <= maxLyr; i++) {
                this.nbrs.add(new ArrayList<>());
            }
        }
    }

    public static class GraphInfo {
        public int topLayer;
        public int nodeCount;
        public List<Integer> nodesPerLayer = new ArrayList<>();
        public List<Integer> edgesPerLayer = new ArrayList<>();
        public List<NodeValue> nodes = new ArrayList<>();
        public List<EdgeValue> edges = new ArrayList<>();
    }

    public record NodeValue(int id, String metadata, String category, int maxLyr) {}
    public record EdgeValue(int src, int dst, int lyr) {}

    private final Map<Integer, Node> G = new ConcurrentHashMap<>();
    private final int M;
    private final int M0;
    private final int ef_build;
    private final float mL;
    private int topLayer = -1;
    private int entryPt = -1;
    private final Random rng = new Random(42);

    public HNSW() {
        this(16, 200);
    }

    public HNSW(int m, int efBuild) {
        this.M = m;
        this.M0 = 2 * m;
        this.ef_build = efBuild;
        this.mL = 1.0f / (float) Math.log(m);
    }

    private int randLevel() {
        double u = rng.nextDouble();
        // Avoid boundary case of u = 0.0
        if (u < 1e-9) u = 1e-9;
        return (int) Math.floor(-Math.log(u) * mL);
    }

    private List<SearchHit> searchLayer(List<Float> q, int ep, int ef, int lyr,
                                        BiFunction<List<Float>, List<Float>, Float> dist,
                                        List<TraversalStep> pathAccumulator) {
        Set<Integer> visited = new HashSet<>();
        // Min-heap for candidates: smallest distance first
        PriorityQueue<SearchHit> cands = new PriorityQueue<>();
        // Max-heap for found results: largest distance first
        PriorityQueue<SearchHit> found = new PriorityQueue<>((a, b) -> Float.compare(b.distance(), a.distance()));

        Node epNode = G.get(ep);
        if (epNode == null) {
            return Collections.emptyList();
        }

        float d0 = dist.apply(q, epNode.item.getEmb());
        visited.add(ep);
        cands.offer(new SearchHit(ep, d0));
        found.offer(new SearchHit(ep, d0));

        while (!cands.isEmpty()) {
            SearchHit curr = cands.poll();
            if (found.size() >= ef && curr.distance() > found.peek().distance()) {
                break;
            }
            Node currNode = G.get(curr.id());
            if (currNode == null || lyr >= currNode.nbrs.size()) {
                continue;
            }
            for (int nid : currNode.nbrs.get(lyr)) {
                if (visited.contains(nid) || !G.containsKey(nid)) {
                    continue;
                }
                visited.add(nid);
                Node nNode = G.get(nid);
                if (nNode == null) continue;

                float nd = dist.apply(q, nNode.item.getEmb());
                if (found.size() < ef || nd < found.peek().distance()) {
                    cands.offer(new SearchHit(nid, nd));
                    found.offer(new SearchHit(nid, nd));
                    if (found.size() > ef) {
                        found.poll();
                    }
                    if (pathAccumulator != null) {
                        pathAccumulator.add(new TraversalStep(curr.id(), nid, lyr, nd));
                    }
                }
            }
        }

        List<SearchHit> res = new ArrayList<>(found);
        Collections.sort(res);
        return res;
    }

    private List<Integer> selectNbrs(List<SearchHit> cands, int maxM) {
        List<Integer> r = new ArrayList<>();
        for (int i = 0; i < Math.min(cands.size(), maxM); i++) {
            r.add(cands.get(i).id());
        }
        return r;
    }

    public synchronized void insert(VectorItem item, BiFunction<List<Float>, List<Float>, Float> dist) {
        int id = item.getId();
        int lvl = randLevel();
        Node node = new Node(item, lvl);
        G.put(id, node);

        if (entryPt == -1) {
            entryPt = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            Node epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<SearchHit> W = searchLayer(item.getEmb(), ep, 1, lc, dist, null);
                if (!W.isEmpty()) {
                    ep = W.get(0).id();
                }
            }
        }

        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<SearchHit> W = searchLayer(item.getEmb(), ep, ef_build, lc, dist, null);
            int maxM = (lc == 0) ? M0 : M;
            List<Integer> sel = selectNbrs(W, maxM);
            node.nbrs.set(lc, sel);

            for (int nid : sel) {
                Node nNode = G.get(nid);
                if (nNode == null) continue;
                while (nNode.nbrs.size() <= lc) {
                    nNode.nbrs.add(new ArrayList<>());
                }
                List<Integer> conn = nNode.nbrs.get(lc);
                conn.add(id);
                if (conn.size() > maxM) {
                    List<SearchHit> ds = new ArrayList<>();
                    for (int c : conn) {
                        Node cNode = G.get(c);
                        if (cNode != null) {
                            float d = dist.apply(nNode.item.getEmb(), cNode.item.getEmb());
                            ds.add(new SearchHit(c, d));
                        }
                    }
                    Collections.sort(ds);
                    conn.clear();
                    for (int i = 0; i < maxM && i < ds.size(); i++) {
                        conn.add(ds.get(i).id());
                    }
                }
            }
            if (!W.isEmpty()) {
                ep = W.get(0).id();
            }
        }

        if (lvl > topLayer) {
            topLayer = lvl;
            entryPt = id;
        }
    }

    public synchronized HNSWKnnResult knn(List<Float> q, int k, int ef, BiFunction<List<Float>, List<Float>, Float> dist) {
        List<TraversalStep> path = new ArrayList<>();
        if (entryPt == -1) {
            return new HNSWKnnResult(Collections.emptyList(), path);
        }
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<SearchHit> W = searchLayer(q, ep, 1, lc, dist, path);
                if (!W.isEmpty()) {
                    ep = W.get(0).id();
                }
            }
        }
        List<SearchHit> W = searchLayer(q, ep, Math.max(ef, k), 0, dist, path);
        if (W.size() > k) {
            W = new ArrayList<>(W.subList(0, k));
        }
        return new HNSWKnnResult(W, path);
    }

    public synchronized void remove(int id) {
        if (!G.containsKey(id)) return;
        for (Node nd : G.values()) {
            for (List<Integer> layer : nd.nbrs) {
                layer.remove(Integer.valueOf(id));
            }
        }
        if (entryPt == id) {
            entryPt = -1;
            for (int nid : G.keySet()) {
                if (nid != id) {
                    entryPt = nid;
                    break;
                }
            }
        }
        G.remove(id);
    }

    public synchronized GraphInfo getInfo() {
        GraphInfo gi = new GraphInfo();
        gi.topLayer = topLayer;
        gi.nodeCount = G.size();
        int maxL = Math.max(topLayer + 1, 1);
        for (int i = 0; i < maxL; i++) {
            gi.nodesPerLayer.add(0);
            gi.edgesPerLayer.add(0);
        }
        for (Map.Entry<Integer, Node> entry : G.entrySet()) {
            int id = entry.getKey();
            Node nd = entry.getValue();
            gi.nodes.add(new NodeValue(id, nd.item.getMetadata(), nd.item.getCategory(), nd.maxLyr));
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) {
                            gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                            gi.edges.add(new EdgeValue(id, nid, lc));
                        }
                    }
                }
            }
        }
        return gi;
    }

    public synchronized int size() {
        return G.size();
    }
}
