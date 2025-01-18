from nicegui import ui

from ui.pages.weaviate_repo import weaviate_repo




with ui.row():
    with ui.card():
        ui.label('Weaviate Repo')
        ui.button('Open', on_click=lambda: ui.navigate.to('/weaviate_repo'))




ui.run()