package com.example.vectordb.engine.db;

import java.util.List;

public class VectorItem {
    private int id;
    private String metadata;
    private String category;
    private List<Float> emb;

    public VectorItem() {}

    public VectorItem(int id, String metadata, String category, List<Float> emb) {
        this.id = id;
        this.metadata = metadata;
        this.category = category;
        this.emb = emb;
    }

    public int getId() {
        return id;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getMetadata() {
        return metadata;
    }

    public void setMetadata(String metadata) {
        this.metadata = metadata;
    }

    public String getCategory() {
        return category;
    }

    public void setCategory(String category) {
        this.category = category;
    }

    public List<Float> getEmb() {
        return emb;
    }

    public void setEmb(List<Float> emb) {
        this.emb = emb;
    }
}
