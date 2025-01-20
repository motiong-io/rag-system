# rag-system

## Requirement
`poetry add git+ssh://git@github.com:motiong-io/reactor.git`

## UI
- weaviate repo 
`python ui/main.py`
NiceGUI will be ready to go on http://localhost:8080, and http://172.22.0.2:8080

## structure
### flow chart
![hybrid rag agent sysyten](assets/img/hybrid_rag_agent.png)
Note: Elastic BM25 was replaced by Weaviate BM25 Search


### data model
![structure overview](assets/img/structure.png)
...
------
update at 20/01/2025 @HoKei2001