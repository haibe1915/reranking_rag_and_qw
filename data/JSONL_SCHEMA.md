# JSONL Data Schema

Tai lieu nay mo ta cau truc cac file JSONL trong thu muc `data/*/processed/`.

## 1. Dinh dang JSONL

- Moi dong la 1 JSON object hop le.
- Khong co dau phay giua cac dong.
- File co the doc theo tung dong (`json.loads(line)`).

## 2. Truong chung (tat ca dataset)

| Truong        | Kieu    | Bat buoc | Mo ta                                       |
| ------------- | ------- | -------- | ------------------------------------------- |
| id            | string  | Co       | ID da chuan hoa cua mau                     |
| title         | string  | Co       | Tieu de/nhom chu de (co the rong voi VIMQA) |
| context       | string  | Co       | Ngu canh phuc vu tra loi                    |
| query         | string  | Co       | Cau hoi                                     |
| ground_truth  | string  | Co       | Dap an chuan                                |
| is_impossible | boolean | Co       | Co/khong co dap an trong context            |

## 3. Truong bo sung theo dataset

### vhealthqa

| Truong | Kieu   | Bat buoc | Mo ta               |
| ------ | ------ | -------- | ------------------- |
| link   | string | Co       | Link nguon bai viet |

### vimqa

| Truong           | Kieu  | Bat buoc | Mo ta                                                         |
| ---------------- | ----- | -------- | ------------------------------------------------------------- |
| supporting_facts | array | Co       | Danh sach fact ho tro, thuong dang `[entity, sentence_index]` |

### vietnamese_rag

| Truong | Kieu   | Bat buoc | Mo ta                                                                                   |
| ------ | ------ | -------- | --------------------------------------------------------------------------------------- |
| source | string | Co       | Nguon raw goc (vd: `modified_data_BKAI`, `modify_legal_corpus`, `rag_viQuAD`, `vi_RAG`) |

## 4. Quy uoc ID hien tai

ID dang dung hien tai:

- Khong co nhom bo sung: `dataset:split:000001`
- Neu co nhom bo sung (neu bat lai): `dataset:group:split:000001`

Vi du:

- `uit_viquad2:test:000001`
- `vhealthqa:train:000123`
- `vimqa:validation:000045`
- `vietnamese_rag:test:000001`

## 5. Vi du record

### 5.1 Mau chung

```json
{
  "id": "uit_viquad2:test:000001",
  "title": "California",
  "context": "...",
  "query": "Bien nao tiep giap voi bang California?",
  "ground_truth": "Thai Binh Duong",
  "is_impossible": false
}
```

### 5.2 vhealthqa

```json
{
  "id": "vhealthqa:test:000001",
  "title": "vhealthqa",
  "context": "https://...",
  "query": "...",
  "ground_truth": "...",
  "is_impossible": false,
  "link": "https://..."
}
```

### 5.3 vimqa

```json
{
  "id": "vimqa:test:000001",
  "title": "",
  "context": "...",
  "query": "...",
  "ground_truth": "dung",
  "is_impossible": false,
  "supporting_facts": [
    ["Diego Maradona", 7],
    ["Boca Juniors", 0]
  ]
}
```

### 5.4 vietnamese_rag

```json
{
  "id": "vietnamese_rag:test:000001",
  "title": "modify_legal_corpus",
  "context": "...",
  "query": "...",
  "ground_truth": "...",
  "is_impossible": false,
  "source": "modify_legal_corpus"
}
```

## 6. Checklist kiem tra nhanh truoc khi train/eval

- File JSONL parse duoc 100% dong.
- Khong thieu truong bat buoc.
- `id` khong trung trong cung 1 split.
- `query` va `ground_truth` khong rong (neu dung cho baseline EM/F1).
