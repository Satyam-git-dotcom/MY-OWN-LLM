package com.example.vectordb.controller;

import com.example.vectordb.engine.algo.DistanceMetrics;
import com.example.vectordb.engine.algo.HNSW;
import com.example.vectordb.engine.algo.TraversalStep;
import com.example.vectordb.engine.db.*;
import com.example.vectordb.engine.ollama.OllamaClient;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.nio.charset.StandardCharsets;
import java.util.*;

@RestController
@CrossOrigin(origins = "*")
@Tag(name = "VectorDB Core API", description = "REST endpoints for HNSW + KD-Tree + Brute Force search indexes, Ollama embeddings, and RAG.")
public class ApiController {

    private final VectorDatabase db;
    private final DocumentDatabase docDB;
    private final OllamaClient ollama;

    public ApiController(VectorDatabase db, DocumentDatabase docDB, OllamaClient ollama) {
        this.db = db;
        this.docDB = docDB;
        this.ollama = ollama;
    }

    // =====================================================================
    //  DTOs
    // =====================================================================

    public record InsertRequest(String metadata, String category, List<Float> embedding) {}
    public record InsertResponse(int id) {}
    public record DocInsertRequest(String title, String text) {}
    public record DocInsertResponse(List<Integer> ids, int chunks, int dims) {}
    public record DocSearchRequest(String question, int k) {}
    public record DocSearchHitResponse(int id, String title, float distance) {}
    public record DocSearchResponse(List<DocSearchHitResponse> contexts) {}
    public record DocAskResponse(String answer, String model, List<DocContextDetail> contexts, int docCount) {}
    public record DocContextDetail(int id, String title, String text, float distance) {}
    public record DocListResponse(int id, String title, String preview, int words) {}

    // =====================================================================
    //  DEMO VECTOR ENDPOINTS
    // =====================================================================

    @Operation(summary = "Search nearest neighbors in 16D demo vector space")
    @GetMapping("/search")
    public ResponseEntity<?> search(
            @Parameter(description = "16D comma-separated float values") @RequestParam("v") String vStr,
            @Parameter(description = "Number of nearest neighbors") @RequestParam(value = "k", defaultValue = "5") int k,
            @Parameter(description = "Distance metric: cosine, euclidean, manhattan, chebyshev, hamming") @RequestParam(value = "metric", defaultValue = "cosine") String metric,
            @Parameter(description = "Search algorithm: hnsw, kdtree, bruteforce") @RequestParam(value = "algo", defaultValue = "hnsw") String algo) {

        List<Float> q = parseVec(vStr);
        if (q.size() != db.dims) {
            return ResponseEntity.badRequest().body(Map.of("error", "need " + db.dims + "D vector"));
        }

        VectorDatabase.SearchOut out = db.search(q, k, metric, algo);
        return ResponseEntity.ok(out);
    }

    @Operation(summary = "Insert a custom 16D vector manually")
    @PostMapping("/insert")
    public ResponseEntity<?> insert(@RequestBody InsertRequest req) {
        if (req.metadata() == null || req.category() == null || req.embedding() == null || req.embedding().size() != db.dims) {
            return ResponseEntity.badRequest().body(Map.of("error", "invalid body or dimension mismatch"));
        }
        int id = db.insert(req.metadata(), req.category(), req.embedding(), DistanceMetrics.getDistanceFn("cosine"));
        return ResponseEntity.ok(new InsertResponse(id));
    }

    @Operation(summary = "Delete a vector from 16D demo indexes by ID")
    @DeleteMapping("/delete/{id}")
    public ResponseEntity<?> delete(@PathVariable("id") int id) {
        boolean ok = db.remove(id);
        return ResponseEntity.ok(Map.of("ok", ok));
    }

    @Operation(summary = "List all vectors currently in 16D indexes")
    @GetMapping("/items")
    public ResponseEntity<?> items() {
        List<VectorItem> items = db.all();
        List<Map<String, Object>> response = new ArrayList<>();
        for (VectorItem item : items) {
            response.add(Map.of(
                    "id", item.getId(),
                    "metadata", item.getMetadata(),
                    "category", item.getCategory(),
                    "embedding", item.getEmb()
            ));
        }
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Run benchmark comparing all 3 search algorithms (HNSW vs KD-Tree vs Brute Force)")
    @GetMapping("/benchmark")
    public ResponseEntity<?> benchmark(
            @RequestParam("v") String vStr,
            @RequestParam(value = "k", defaultValue = "5") int k,
            @RequestParam(value = "metric", defaultValue = "cosine") String metric) {

        List<Float> q = parseVec(vStr);
        if (q.size() != db.dims) {
            return ResponseEntity.badRequest().body(Map.of("error", "need " + db.dims + "D vector"));
        }
        VectorDatabase.BenchOut b = db.benchmark(q, k, metric);
        return ResponseEntity.ok(Map.of(
                "bruteforceUs", b.bruteforceUs(),
                "kdtreeUs", b.kdtreeUs(),
                "hnswUs", b.hnswUs(),
                "itemCount", b.itemCount()
        ));
    }

    @Operation(summary = "Retrieve HNSW graph layered nodes, edges and levels statistics")
    @GetMapping("/hnsw-info")
    public ResponseEntity<?> hnswInfo() {
        HNSW.GraphInfo gi = db.hnswInfo();
        return ResponseEntity.ok(gi);
    }

    @Operation(summary = "Get 16D database stats")
    @GetMapping("/stats")
    public ResponseEntity<?> stats() {
        return ResponseEntity.ok(Map.of(
                "count", db.size(),
                "dims", db.dims,
                "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
                "metrics", List.of("euclidean", "cosine", "manhattan", "chebyshev", "hamming")
        ));
    }

    // =====================================================================
    //  DOCUMENT & RAG ENDPOINTS
    // =====================================================================

    @Operation(summary = "Chunk, embed via Ollama and store document text in 768D HNSW")
    @PostMapping("/doc/insert")
    public ResponseEntity<?> docInsert(@RequestBody DocInsertRequest req) {
        if (req.title() == null || req.text() == null || req.title().trim().isEmpty() || req.text().trim().isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "need title and text"));
        }

        List<String> chunks = TextChunker.chunkText(req.text(), 250, 30);
        List<Integer> ids = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            List<Float> emb = ollama.embed(chunks.get(i));
            if (emb.isEmpty()) {
                return ResponseEntity.internalServerError().body(Map.of("error",
                        "Ollama unavailable. Run: ollama pull nomic-embed-text"));
            }
            String chunkTitle = chunks.size() > 1
                    ? req.title() + " [" + (i + 1) + "/" + chunks.size() + "]"
                    : req.title();
            ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
        }

        return ResponseEntity.ok(new DocInsertResponse(ids, chunks.size(), docDB.getDims()));
    }

    @Operation(summary = "Upload and embed a raw text or markdown file (.txt, .md)")
    @PostMapping(value = "/doc/upload", consumes = "multipart/form-data")
    public ResponseEntity<?> docUpload(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "title", required = false) String customTitle) {

        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "uploaded file is empty"));
        }
        try {
            String text = new String(file.getBytes(), StandardCharsets.UTF_8);
            String title = (customTitle == null || customTitle.trim().isEmpty())
                    ? file.getOriginalFilename()
                    : customTitle;

            return docInsert(new DocInsertRequest(title, text));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(Map.of("error", "Failed to process file upload: " + e.getMessage()));
        }
    }

    @Operation(summary = "Delete stored document chunk by ID")
    @DeleteMapping("/doc/delete/{id}")
    public ResponseEntity<?> docDelete(@PathVariable("id") int id) {
        boolean ok = docDB.remove(id);
        return ResponseEntity.ok(Map.of("ok", ok));
    }

    @Operation(summary = "List all stored document chunks with previews")
    @GetMapping("/doc/list")
    public ResponseEntity<?> docList() {
        List<DocItem> docs = docDB.all();
        List<DocListResponse> response = new ArrayList<>();
        for (DocItem d : docs) {
            String preview = d.getText().length() > 120
                    ? d.getText().substring(0, 120) + "…"
                    : d.getText();
            int words = d.getText().split("\\s+").length;
            response.add(new DocListResponse(d.getId(), d.getTitle(), preview, words));
        }
        return ResponseEntity.ok(response);
    }

    @Operation(summary = "Fast retrieval search returning context metadata lists")
    @PostMapping("/doc/search")
    public ResponseEntity<?> docSearch(@RequestBody DocSearchRequest req) {
        if (req.question() == null || req.question().trim().isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "need question"));
        }
        List<Float> qEmb = ollama.embed(req.question());
        if (qEmb.isEmpty()) {
            return ResponseEntity.internalServerError().body(Map.of("error", "Ollama unavailable"));
        }
        List<DocSearchHit> hits = docDB.search(qEmb, req.k() <= 0 ? 3 : req.k(), 0.7f);
        List<DocSearchHitResponse> response = new ArrayList<>();
        for (DocSearchHit hit : hits) {
            response.add(new DocSearchHitResponse(hit.docItem().getId(), hit.docItem().getTitle(), hit.distance()));
        }
        return ResponseEntity.ok(new DocSearchResponse(response));
    }

    @Operation(summary = "RAG Pipeline: embed question -> retrieve matching document context -> generate local LLM answer")
    @PostMapping("/doc/ask")
    public ResponseEntity<?> docAsk(@RequestBody DocSearchRequest req) {
        if (req.question() == null || req.question().trim().isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "need question"));
        }
        int k = req.k() <= 0 ? 3 : req.k();

        // 1. Embed question
        List<Float> qEmb = ollama.embed(req.question());
        if (qEmb.isEmpty()) {
            return ResponseEntity.internalServerError().body(Map.of("error", "Ollama unavailable"));
        }

        // 2. Retrieve context chunks
        List<DocSearchHit> hits = docDB.search(qEmb, k, 0.7f);

        // 3. Build Prompt
        StringBuilder ctx = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            ctx.append("[").append(i + 1).append("] ").append(hits.get(i).docItem().getTitle()).append(":\n")
               .append(hits.get(i).docItem().getText()).append("\n\n");
        }
        String prompt =
            "You are a helpful assistant. Answer the user's question directly. " +
            "Use the provided context if it contains relevant information. " +
            "If it doesn't, just use your own general knowledge. " +
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. " +
            "Just answer the question naturally.\n\n" +
            "Context:\n" + ctx.toString() +
            "Question: " + req.question() + "\n\n" +
            "Answer:";

        // 4. Generate response
        String answer = ollama.generate(prompt);

        // 5. Build DTO Response
        List<DocContextDetail> contexts = new ArrayList<>();
        for (DocSearchHit hit : hits) {
            contexts.add(new DocContextDetail(
                    hit.docItem().getId(),
                    hit.docItem().getTitle(),
                    hit.docItem().getText(),
                    hit.distance()
            ));
        }

        return ResponseEntity.ok(new DocAskResponse(answer, ollama.genModel, contexts, docDB.size()));
    }

    @Operation(summary = "Get Ollama and active databases diagnostic status")
    @GetMapping("/status")
    public ResponseEntity<?> getStatus() {
        boolean available = ollama.isAvailable();
        return ResponseEntity.ok(Map.of(
                "ollamaAvailable", available,
                "embedModel", ollama.embedModel,
                "genModel", ollama.genModel,
                "docCount", docDB.size(),
                "docDims", docDB.getDims(),
                "demoDims", db.dims,
                "demoCount", db.size()
        ));
    }

    // =====================================================================
    //  HELPERS
    // =====================================================================

    private List<Float> parseVec(String s) {
        if (s == null || s.trim().isEmpty()) {
            return Collections.emptyList();
        }
        List<Float> list = new ArrayList<>();
        String[] tokens = s.split(",");
        for (String t : tokens) {
            try {
                list.add(Float.parseFloat(t.trim()));
            } catch (NumberFormatException ignored) {}
        }
        return list;
    }
}
