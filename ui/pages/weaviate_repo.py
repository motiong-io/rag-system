from nicegui import ui

from ui.utils.manage_weaviate import check_weaviate_connection, check_collections

@ui.page('/weaviate_repo')
def weaviate_repo():
    ui.page_title('Weaviate Repo')

    
    weavicate_connection = check_weaviate_connection()
    collections = check_collections()
    collection_name_list = sorted([collection.get('collection_name') for collection in collections])

    # main space
    ui.markdown("# Weaviate Repo")
    main_space = ui.column().style('height: 100vh; overflow-y: auto;')

    # header
    with ui.header().classes(replace='row items-center') as header:
        ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')
        ui.button(on_click=ui.navigate.back,icon='home').props('flat color=white')

    # left drawer
    with ui.left_drawer().classes('bg-blue-100') as left_drawer:
        ui.markdown('#### Connection Status')
        if weavicate_connection:
            ui.markdown(' 🟢 Weaviate is connected')
        else:  
            ui.markdown(' 🔴 Weaviate is not connected')

        ui.markdown('#### Weaviate collections')

        displayed_collection = {'collection_name': 'Click one collection to see', 'properties': [{'property_name': 'property_name', 'data_type': 'data_type'}]}

        @ui.refreshable
        def display_collection_info() -> None:
            ui.label(displayed_collection['collection_name']).style('color: green; font-size: 20px;')
            ui.table(rows=displayed_collection['properties'])
        
        def update_displayed_collection(collection_name):
            ui.notify(f"You clicked on collection: {collection_name}")
            for collection in collections:
                if collection.get('collection_name') == collection_name:
                    displayed_collection['collection_name'] = collection_name
                    displayed_collection['properties'] = collection.get('properties')
                    break
            display_collection_info.refresh()
        
        
        with ui.column():
            for collection_name in collection_name_list:
                ui.button(collection_name, on_click=lambda collection=collection_name: update_displayed_collection(collection),color='').style('width: 100%;')
        
        ui.markdown('#### Collection Information')
        display_collection_info()

        
        
