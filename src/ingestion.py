"""Document parsing"""
import base64
import asyncio
import nest_asyncio
from llama_parse import LlamaParse
nest_asyncio.apply()


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
            premium_mode=True
        )


    def parse_docs(self):
        """Parse PDF into structured JSON/Markdown"""
        return self.parser.get_json_result(self.pdf_path)


    def encode_image(self, img_path):
        """Encode image to base64 string"""
        with open(img_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')


    async def async_get_images(self, obj):
        """Asynchronously get images using LlamaParse"""
        return await self.parser.aget_images(obj, download_path=self.image_download_path)


    def process_images(self, obj):
        """Download images from PDF and encode as base64"""
        encoded_images = []

        # Get the current event loop or create a new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            # Run the async operation in the event loop
            image_dicts = loop.run_until_complete(self.async_get_images(obj))

            # Process images
            for image in image_dicts:
                if image['path'].endswith("png"):
                    encoded_images.append(self.encode_image(image['path']))

        except Exception as e:
            print(f"Error processing images: {str(e)}")
            return []

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
                heading_md = item.get('md', item.get('value', ''))
                heading_md = ' '.join(heading_md.split())
                combined_text = heading_md

                if i + 1 < len(items) and items[i + 1]['type'] == 'text':
                    text_md = items[i + 1].get('md', items[i + 1].get('value', ''))
                    text_md = ' '.join(text_md.split())
                    combined_text = f"{heading_md}\n\n{text_md}"
                    i += 2
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
                text_list.extend(page_text)
            if page_tables:
                table_list.extend(page_tables)

        return text_list, table_list, images_list
