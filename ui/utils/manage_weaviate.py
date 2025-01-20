
import weaviate
from weaviate.classes.query import Filter
from app.config import env
from typing import List

def check_weaviate_connection():
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=env.weaviate_url,
        auth_credentials=weaviate.classes.init.Auth.api_key(env.weaviate_api_key),
    ) as client:
        if client.is_ready():
            return True
        else:  
            return False


def check_collections()->List[dict]:
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=env.weaviate_url,
        auth_credentials=weaviate.classes.init.Auth.api_key(env.weaviate_api_key)) as client:
            all_collections = client.collections.list_all(simple=True)

    result = []
    for collection, config in all_collections.items():
        collection_info = {"collection_name": collection, "properties": []}
        for prop in config.properties:
            collection_info['properties'].append({
                 'property_name': prop.name,
                 'data_type': prop.data_type.value})
        result.append(collection_info)
    return sorted(result, key=lambda x: x.get('collection_name', ''))


def aggregate_collection(collection_name:str):
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=env.weaviate_url,
        auth_credentials=weaviate.classes.init.Auth.api_key(env.weaviate_api_key)) as client:
        collection = client.collections.get(collection_name)
        total_count = collection.aggregate.over_all(total_count=True).total_count
        return total_count
    
def aggregate_objects_by_property(collection_name:str,property_name:str, property_value:str):
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=env.weaviate_url,
        auth_credentials=weaviate.classes.init.Auth.api_key(env.weaviate_api_key)
        )as client:

        collection = client.collections.get(collection_name)
        response = collection.aggregate.over_all(
            filters=Filter.by_property(property_name).equal(property_value),
        )
        return response.total_count
    

