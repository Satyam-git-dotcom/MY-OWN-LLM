package com.example.vectordb.engine.ollama;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.springframework.stereotype.Component;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Component
public class OllamaClient {

    private final String host = "127.0.0.1";
    private final int port = 11434;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final HttpClient httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(3))
            .build();

    public String embedModel = "nomic-embed-text";
    public String genModel = "llama3.2";

    public boolean isAvailable() {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/tags"))
                    .timeout(Duration.ofSeconds(2))
                    .GET()
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            return response.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    public List<Float> embed(String text) {
        try {
            ObjectNode payload = objectMapper.createObjectNode();
            payload.put("model", embedModel);
            payload.put("prompt", text);
            String jsonPayload = objectMapper.writeValueAsString(payload);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/embeddings"))
                    .timeout(Duration.ofSeconds(30))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                return Collections.emptyList();
            }

            JsonNode root = objectMapper.readTree(response.body());
            JsonNode embeddingNode = root.get("embedding");
            if (embeddingNode != null && embeddingNode.isArray()) {
                List<Float> result = new ArrayList<>(embeddingNode.size());
                for (JsonNode val : embeddingNode) {
                    result.add((float) val.asDouble());
                }
                return result;
            }
        } catch (Exception e) {
            // Log or handle exception
            System.err.println("Error calling Ollama embeddings: " + e.getMessage());
        }
        return Collections.emptyList();
    }

    public String generate(String prompt) {
        try {
            ObjectNode payload = objectMapper.createObjectNode();
            payload.put("model", genModel);
            payload.put("prompt", prompt);
            payload.put("stream", false);
            String jsonPayload = objectMapper.writeValueAsString(payload);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/generate"))
                    .timeout(Duration.ofSeconds(180)) // LLM generation can be slow
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(jsonPayload))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                return "ERROR: Ollama returned status " + response.statusCode();
            }

            JsonNode root = objectMapper.readTree(response.body());
            JsonNode responseNode = root.get("response");
            if (responseNode != null) {
                return responseNode.asText();
            }
        } catch (Exception e) {
            System.err.println("Error calling Ollama generate: " + e.getMessage());
        }
        return "ERROR: Ollama unavailable. Run: ollama serve";
    }
}
