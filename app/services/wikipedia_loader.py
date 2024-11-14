import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import re
from app.model.document_json import DocumentJSON
from app.utils.md5hash import md5hash

class WikipediaLoader:
    """
        A class to load and parse content from a Wikipedia page.
        - url: The URL of the Wikipedia page.
        - filename: The name of the markdown file to save the parsed content.
    """
    def __init__(self, url):
        self.url = url
        self.uuid = md5hash(url)
        self.soup = self.get_soup()
        self.title = self.get_title()
        self.body_content = self.get_body_content()
        self.markdown_content = self.parse_content()
    
    def get_soup(self):
        """Return a BeautifulSoup object for the given url."""
        response = requests.get(self.url)
        return BeautifulSoup(response.content, 'html.parser')

    def get_title(self):
        """Return the title of the Wikipedia page."""
        title = self.soup.find(id="firstHeading")
        return title.contents[0].getText()

    def get_body_content(self):
        """Return the body content of the Wikipedia page."""
        body_content = self.soup.find(id="bodyContent")
        my_content_text = body_content.find(id="mw-content-text")
        content_div = my_content_text.find('div', class_=['mw-content-ltr', 'mw-parser-output'])
        # print(content_div)
        return content_div

    def parse_content(self):
        """Parse the content of the Wikipedia page."""
        def tag_to_markdown(tag):
            for a in tag.find_all('a'):
                a.replace_with(a.get_text())
            return md(str(tag)) if tag else ''
    
        def consolidate_newlines(text):
            # Use a regular expression to replace multiple newlines with a single newline
            consolidated_text = re.sub(r'\n\s*\n+', '\n\n', text).strip()
            return consolidated_text
        
        def replace_steps(text: str) -> str:
            # 使用 re.sub 去除引用标记 `\\[1]`, `\\[2]`, 等数字标记
            result = re.sub(r'\\\[\d+\]', '', text)
            # 用正则表达式移除像 `\\[A]`, `\\[B]`, `\\[C]` 等字母标记
            result = re.sub(r'\\\[[A-Z]\]', '', result)
            # 替换掉 \u00a0
            result = result.replace('\u00a0', ' ')
            result = result.replace(r'\-', '-')
            return result
        
        markdown_content = ''
        # Iterate over all direct children of the main content area
        for element in self.body_content.find_all():
            element_id = element.get('id')
            if element_id == 'References' or element_id=='Citations' or element_id=='See_also':  # Stop at the references section
                break

            if element.get('role') == 'navigation': 
                continue

            if element.name == 'p':  # For paragraphs
                markdown_content += tag_to_markdown(element) + '\n'
            elif element.name == 'table':  # For tables
                markdown_content += tag_to_markdown(element) + '\n'
            # Add conditions for other elements you might want to include
            elif element.name == 'h2' or element.name == 'h3':  # For headings
                markdown_content += tag_to_markdown(element) + '\n'
            elif element.name == 'ul':  # For unordered lists
                markdown_content += tag_to_markdown(element) + '\n'
            elif element.name == 'ol':  # For ordered lists
                markdown_content += tag_to_markdown(element) + '\n'
        
        return replace_steps(consolidate_newlines(markdown_content))
    
    def save_markdown(self, filename):
        """Save the parsed content to a markdown file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f'# {self.title}\n\n')
            f.write(self.markdown_content)
            print(f"Markdown content saved to {filename}")

    def save_to_document_json(self, filename):
        """Return a DocumentJSON object with the parsed content."""
        with open(filename, 'w', encoding='utf-8') as f:
            doc = DocumentJSON(self.markdown_content, self.title, self.url)
            doc.save_json(filename)
            print(f"Document saved to {filename}")  


def main():
    url="https://en.wikipedia.org/wiki/List_of_tallest_buildings_in_New_York_City"
    # url="https://en.wikipedia.org/wiki/Jane_Eyre"
    loader = WikipediaLoader(url)
    loader.save_to_document_json("tallest_buildings_nyc.json")
    loader.save_markdown("tallest_buildings_nyc.md")

if __name__ == "__main__":
    main()