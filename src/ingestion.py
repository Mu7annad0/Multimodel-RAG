"""Document parsing"""
import base64
from llama_parse import LlamaParse


class Parser:
    """Parse PDF documents into structured text, tables, and images."""
    def __init__(self, pdf_path: str, image_download_path: str="../images"):
        """
        Args:
            pdf_path: Path to input PDF file.
            image_download_path: Directory to save extracted images. Defaults to "../images".

        Returns (from process()):
            tuple: (base64_images, cleaned_texts, markdown_tables)
        """
        super().__init__()
        self.pdf_path = pdf_path
        self.image_download_path = image_download_path
        self.parser = LlamaParse(
            result_type="markdown",
            # invalidate_cache=True
        )


    def parse_docs(self):
        """Parse PDF into structured JSON/Markdown"""
        return self.parser.get_json_result(self.pdf_path)


    def encode_image(self, img_path):
        """Encode image to base64 string"""
        with open(img_path,'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')


    def process_images(self, obj):
        """Download images from PDF and encode as base64"""
        encoded_images = []
        image_dicts = self.parser.get_images(obj, download_path=self.image_download_path)
        for image in image_dicts:
            encoded_images.append(self.encode_image(image['path']))
        return encoded_images


    def extract_content(self, items):
        """Extract and organize text/tables for each page"""
        table_list = []
        text_list = []
        i = 0
        while i < len(items):
            item = items[i]
            if item['type'] == 'table':
                table_list.append(item.get('md', item.get('value', '')))
                i += 1
            elif item['type'] == 'heading':
                # Process heading and check if the next item is text to merge
                heading_md = item.get('md', item.get('value', ''))
                heading_md = ' '.join(heading_md.split())
                combined_text = heading_md

                # If the heading is immediately followed by a text item, merge them.
                if i + 1 < len(items) and items[i + 1]['type'] == 'text':
                    text_md = items[i + 1].get('md', items[i + 1].get('value', ''))
                    text_md = ' '.join(text_md.split())
                    combined_text = f"{heading_md}\n\n{text_md}"
                    i += 2  # Skip the merged text item
                else:
                    i += 1

                text_list.append(combined_text)
            elif item['type'] == 'text':
                # For standalone text (not immediately following a heading)
                text_value = item.get('md', item.get('value', ''))
                normalized_text = ' '.join(text_value.split())

                # Append this text to the previous text_list item (if it exists)
                if text_list:
                    text_list[-1] += "\n\n" + normalized_text
                else:
                    # If there's no previous text, add it as a new item.
                    text_list.append(normalized_text)
                i += 1
            else:
                i += 1

        return text_list, table_list


    def process(self):
        """Execute full parsing pipeline (main entry point)"""
        md_json_objs = self.parse_docs()
        images_list = self.process_images(md_json_objs)

        table_list = []
        text_list = []

        # Iterate through all pages
        pages = md_json_objs[0]['pages']

        for page in pages:
            items = page['items']
            page_text, page_tables = self.extract_content(items)
            if page_text:
                text_list.append(page_text)
            if page_tables:
                table_list.append(page_tables)

        return images_list, text_list, table_list
