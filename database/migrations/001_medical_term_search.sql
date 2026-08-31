-- 의료용어 정규화용 PostgreSQL 스키마와 검색 함수
-- 작성자: 김진우

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE OR REPLACE FUNCTION normalize_medical_search_text(value text)
RETURNS text
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT regexp_replace(lower(coalesce(value, '')), '[^0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ]+', '', 'g');
$$;

CREATE TABLE IF NOT EXISTS medical_terms (
    canonical_key text PRIMARY KEY,
    canonical_name text NOT NULL,
    term_type text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS medical_term_aliases (
    alias_id bigserial PRIMARY KEY,
    canonical_key text NOT NULL REFERENCES medical_terms(canonical_key) ON DELETE CASCADE,
    alias_display text NOT NULL,
    alias_normalized text NOT NULL,
    alias_initials text NOT NULL DEFAULT '',
    priority integer NOT NULL DEFAULT 0,
    source_type text NOT NULL DEFAULT 'SOURCE_DATA',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (canonical_key, alias_normalized)
);

CREATE INDEX IF NOT EXISTS ix_medical_term_alias_normalized_trgm
    ON medical_term_aliases USING gin (alias_normalized gin_trgm_ops);
CREATE INDEX IF NOT EXISTS ix_medical_term_alias_initials
    ON medical_term_aliases (alias_initials)
    WHERE alias_initials <> '';

CREATE OR REPLACE FUNCTION search_medical_terms(
    input_query text,
    result_limit integer DEFAULT 8
)
RETURNS TABLE (
    canonical_key text,
    canonical_name text,
    term_type text,
    matched_alias text,
    match_score double precision,
    match_kind text,
    match_priority integer
)
LANGUAGE sql
STABLE
AS $$
    WITH query_value AS (
        SELECT normalize_medical_search_text(input_query) AS normalized
    ),
    candidates AS (
        SELECT
            term.canonical_key,
            term.canonical_name,
            term.term_type,
            alias.alias_display AS matched_alias,
            CASE
                WHEN alias.alias_normalized = query_value.normalized THEN 1.0
                WHEN alias.alias_normalized LIKE '%' || query_value.normalized || '%'
                  OR query_value.normalized LIKE '%' || alias.alias_normalized || '%' THEN 0.96
                WHEN alias.alias_initials <> ''
                  AND alias.alias_initials = query_value.normalized THEN 0.97
                ELSE greatest(
                    similarity(alias.alias_normalized, query_value.normalized),
                    word_similarity(query_value.normalized, alias.alias_normalized)
                )
            END AS match_score,
            CASE
                WHEN alias.alias_normalized = query_value.normalized THEN 'exact'
                WHEN alias.alias_normalized LIKE '%' || query_value.normalized || '%'
                  OR query_value.normalized LIKE '%' || alias.alias_normalized || '%' THEN 'substring'
                WHEN alias.alias_initials <> ''
                  AND alias.alias_initials = query_value.normalized THEN 'initials'
                ELSE 'fuzzy'
            END AS match_kind,
            alias.priority AS match_priority
        FROM medical_term_aliases AS alias
        JOIN medical_terms AS term USING (canonical_key)
        CROSS JOIN query_value
        WHERE query_value.normalized <> ''
          AND (
              alias.alias_normalized = query_value.normalized
              OR alias.alias_normalized LIKE '%' || query_value.normalized || '%'
              OR query_value.normalized LIKE '%' || alias.alias_normalized || '%'
              OR alias.alias_initials = query_value.normalized
              OR similarity(alias.alias_normalized, query_value.normalized) >= 0.35
              OR word_similarity(query_value.normalized, alias.alias_normalized) >= 0.35
          )
    )
    SELECT *
    FROM candidates
    ORDER BY match_score DESC, match_priority DESC, length(matched_alias) DESC, canonical_key
    LIMIT greatest(1, least(coalesce(result_limit, 8), 20));
$$;
