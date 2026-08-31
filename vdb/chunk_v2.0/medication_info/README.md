# 복약정보 컬렉션

- 작성자: 김진우
- Pinecone namespace: `medication_info`
- 원천: 식품의약품안전처 의약품개요정보(e약은요) OpenAPI
- 적재 패키지: `data/eyak/eyak`
- 임베딩: `jhgan/ko-sroberta-multitask`, 768차원, cosine

사전 계산된 벡터는 `vdb/script/manage_pinecone.py ingest-precomputed` 명령으로
적재한다. 이 폴더는 애플리케이션이 `medication_info` collection을 등록하도록
유지한다.
