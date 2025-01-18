import asyncio
from nicegui import ui
from nicegui.events import GenericEventArguments


from ui.utils.manage_weaviate import check_weaviate_connection, check_collections, aggregate_collection
from ui.utils.benchmark_dataset import load_data,get_page_data,string_to_list,get_docs_data

@ui.page('/weaviate_repo')
def weaviate_repo():
    ui.page_title('Weaviate Repo')

    
    weavicate_connection = check_weaviate_connection()
    collections = check_collections()
    collection_name_list = sorted([collection.get('collection_name') for collection in collections])

    count, page_data_index_range = load_data()

    top_space = ui.column().style('width: 100%')#.classes('border border-blue-100')

    with top_space:
        ui.markdown("## **Weaviate Repo**")
        ui.markdown("This page displays the collections and their properties in Weaviate.")
        ui.markdown("You can switch between collections to view their properties and total records count.")


    # main space
    main_space = ui.row(align_items='start').style('width: 100%; overflow-y: auto;')#.classes('border border-blue-100')
    with main_space:
        ui.space()
        left_main_space = ui.column().style('width:100%;padding:5px')#.classes('border border-blue-100')
        # middle_main_space = ui.column().style('width:1%').classes('border border-blue-100')
        # right_main_space = ui.column().style('width:30%').classes('border border-blue-100')
        with ui.dialog() as dialog,ui.card():
            ui.label('Wiki Docs')

    # header
    with ui.header().classes(replace='row items-center') as header:
        ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('flat color=white')
        ui.button(on_click=ui.navigate.back,icon='home').props('flat color=white')

    # left drawer
    with ui.left_drawer().classes('bg-blue-100') as left_drawer:
        ui.markdown('#### **Connection Status**')
        if weavicate_connection:
            ui.markdown(' 🟢 Weaviate is connected')
        else:  
            ui.markdown(' 🔴 Weaviate is not connected')

        ui.markdown('#### **Weaviate collections**')

        displayed_collection = collections[3] if collections else {'collection_name': 'No collection', 'properties': []}

        @ui.refreshable
        def display_collection_info() -> None:

            def display_wikidocs(uuid:str):
                dialog.clear()
                with dialog, ui.card():
                    ui.label(uuid)
                dialog.open()

            def display_dataset_page(page_index:int):
                left_main_space.clear()
                page_data = get_page_data(page_data_index_range[page_index])
                with left_main_space:
                    for row in page_data['rowData']:
                        wiki_links=string_to_list(row['wiki_links'])
                        with ui.card().style('width:100%;'):
                            with ui.row().style('width:100%'):
                                ui.label(f"{row['Unnamed: 0']}").style('font-size: 20px;width:5%')
                                with ui.column().style('width:30%'):
                                    ui.label(f"{row['Prompt']}")
                                    ui.label(f"{row['Answer']}")
                                    ui.label(f"Wiki Links: {len(wiki_links)}")
                                ui.space()
                                ui.aggrid(options=get_docs_data(wiki_links),html_columns=[1]).style('width:60%; height: 200px;').on('cellClicked', lambda e: display_wikidocs(e.args['data']['uuid']))

                                    # ui.table(get_docs_data(displayed_collection['collection_name'], wiki_links)).on('click', lambda e:print(e.sender))
                                          #, on_click=lambda e:print(e.sender)).style('width:10%')

                        # ui.aggrid(options=page_data).style('width:100%;').on('cellClicked', lambda e: display_wikidocs(e.args['data']))



            top_space.clear()
            left_main_space.clear()
            # right_main_space.clear()
            # ui.label(displayed_collection['collection_name']).style('color: green; font-size: 20px;')
            ui.table(rows=displayed_collection['properties'])
        
            with top_space:
                ui.markdown(f'## **{displayed_collection["collection_name"]}** - benchmark dataset')
                with ui.row().style('width: 100%; padding: 10px;'):
                    ui.label(f'Chunk records count: {aggregate_collection(displayed_collection["collection_name"])}')
                    ui.label(f'Dataset rows count: {count}')
                    ui.label(f'Total pages: {len(page_data_index_range)}')
                    # ui.label(f'index rage: {page_data_index_range}')
                    ui.space()
                    ui.pagination(1, len(page_data_index_range),direction_links=True,on_change=lambda e: display_dataset_page(e.value-1))


            display_dataset_page(0)

                
        def update_displayed_collection(collection_name):
            ui.notify(f"Switch to: {collection_name}")
            for collection in collections:
                if collection.get('collection_name') == collection_name:
                    displayed_collection['collection_name'] = collection_name
                    displayed_collection['properties'] = collection.get('properties')
                    break
            display_collection_info.refresh()

        
        with ui.column():
            for collection_name in collection_name_list:
                ui.button(collection_name, on_click=lambda collection=collection_name: update_displayed_collection(collection),color='').style('width: 100%;')
        
        ui.markdown('#### **Collection Properties**')
        display_collection_info()

    
        
        
