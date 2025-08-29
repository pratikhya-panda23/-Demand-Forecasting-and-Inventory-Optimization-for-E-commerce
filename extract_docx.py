import docx
import sys

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return f"Error extracting text: {str(e)}"

if __name__ == "__main__":
    file_path = "c:\\Users\\adity\\OneDrive\\Desktop\\Coreline Projects\\Data Science 2.docx"
    text = extract_text_from_docx(file_path)
    print(text)
    
    # Also save to a text file for easier viewing
    with open("c:\\Users\\adity\\OneDrive\\Desktop\\Coreline Projects\\extracted_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("\nText has been saved to 'extracted_content.txt'")