-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create ENUM for unit types
DO $$ BEGIN
    CREATE TYPE unit_type_enum AS ENUM (
        'chapter', 'section', 'article', 
        'paragraph', 'paragraph_ext', 
        'def_group', 'def_item'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 3. Create the Main Table
CREATE TABLE IF NOT EXISTS law_units (
    id VARCHAR(100) PRIMARY KEY,
    
    -- Hierarchy
    parent_id VARCHAR(100) REFERENCES law_units(id),
    logical_base_id VARCHAR(100) REFERENCES law_units(id),
    
    -- Content
    content TEXT NOT NULL,
    unit_type unit_type_enum NOT NULL,
    metadata JSONB DEFAULT '{}',
    
    -- Embeddings (1536 dim for OpenAI)
    embedding vector(1536)
);

-- 4. Create Indexes
CREATE INDEX IF NOT EXISTS idx_law_units_parent ON law_units(parent_id);
-- HNSW Index for fast vector search
CREATE INDEX IF NOT EXISTS idx_law_units_embedding 
ON law_units USING hnsw (embedding vector_cosine_ops);