package com.example.vectordb.engine.db;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class TextChunker {

    public static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        if (text == null || text.trim().isEmpty()) {
            return Collections.emptyList();
        }
        
        // Split by any whitespace
        String[] words = text.trim().split("\\s+");
        if (words.length == 0) {
            return Collections.emptyList();
        }
        
        if (words.length <= chunkWords) {
            return List.of(text);
        }
        
        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        if (step <= 0) {
            step = 1; // Prevent infinite loop if overlap is larger than chunk
        }
        
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            StringBuilder sb = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) {
                    sb.append(" ");
                }
                sb.append(words[j]);
            }
            chunks.add(sb.toString());
            if (end == words.length) {
                break;
            }
        }
        return chunks;
    }
}
