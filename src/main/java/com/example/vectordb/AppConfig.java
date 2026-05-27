package com.example.vectordb;

import com.example.vectordb.engine.db.DocumentDatabase;
import com.example.vectordb.engine.db.VectorDatabase;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AppConfig {

    @Bean
    public VectorDatabase vectorDatabase() {
        return new VectorDatabase(16); // 16-D Demo vectors
    }

    @Bean
    public DocumentDatabase documentDatabase() {
        return new DocumentDatabase(); // Dynamic dimensions based on Ollama embeddings (768D)
    }
}
