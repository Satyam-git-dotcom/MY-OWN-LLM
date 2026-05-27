package com.example.vectordb.engine.algo;

import java.util.List;
import java.util.function.BiFunction;

public class DistanceMetrics {

    public static float euclidean(List<Float> a, List<Float> b) {
        float sum = 0;
        int size = Math.min(a.size(), b.size());
        for (int i = 0; i < size; i++) {
            float diff = a.get(i) - b.get(i);
            sum += diff * diff;
        }
        return (float) Math.sqrt(sum);
    }

    public static float cosine(List<Float> a, List<Float> b) {
        float dotProduct = 0;
        float normA = 0;
        float normB = 0;
        int size = Math.min(a.size(), b.size());
        for (int i = 0; i < size; i++) {
            float valA = a.get(i);
            float valB = b.get(i);
            dotProduct += valA * valB;
            normA += valA * valA;
            normB += valB * valB;
        }
        if (normA < 1e-9f || normB < 1e-9f) {
            return 1.0f; // Return maximum distance (dissimilar)
        }
        return 1.0f - (dotProduct / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB)));
    }

    public static float manhattan(List<Float> a, List<Float> b) {
        float sum = 0;
        int size = Math.min(a.size(), b.size());
        for (int i = 0; i < size; i++) {
            sum += Math.abs(a.get(i) - b.get(i));
        }
        return sum;
    }

    public static float chebyshev(List<Float> a, List<Float> b) {
        float max = 0;
        int size = Math.min(a.size(), b.size());
        for (int i = 0; i < size; i++) {
            float diff = Math.abs(a.get(i) - b.get(i));
            if (diff > max) {
                max = diff;
            }
        }
        return max;
    }

    public static float hamming(List<Float> a, List<Float> b) {
        float count = 0;
        int size = Math.min(a.size(), b.size());
        for (int i = 0; i < size; i++) {
            // Count mismatches (with tolerance for floating points)
            if (Math.abs(a.get(i) - b.get(i)) > 0.01f) {
                count++;
            }
        }
        return count;
    }

    public static BiFunction<List<Float>, List<Float>, Float> getDistanceFn(String metricName) {
        if (metricName == null) {
            return DistanceMetrics::euclidean;
        }
        switch (metricName.toLowerCase()) {
            case "cosine":
                return DistanceMetrics::cosine;
            case "manhattan":
                return DistanceMetrics::manhattan;
            case "chebyshev":
                return DistanceMetrics::chebyshev;
            case "hamming":
                return DistanceMetrics::hamming;
            case "euclidean":
            default:
                return DistanceMetrics::euclidean;
        }
    }
}
