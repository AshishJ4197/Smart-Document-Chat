import fitz  # PyMuPDF

# File paths
input_pdf = "../frontend/class10phy.pdf"
output_pdf = "../frontend2/public/highlighted_output.pdf"

# Open the PDF
doc = fitz.open(input_pdf)

# Page 2 (index 1)
page = doc[1]

# List of starting phrases (can be partial)
target_phrases = [
    "Let us recall these laws",
    "(i)",
    "The angle of incidence is equal to the angle of reflection, and",
    "(ii)",
    "The incident ray, the normal to the mirror"
]

# Fetch all lines from the page
lines = page.get_text("text").split('\n')

# Go through each phrase and find partial match
for phrase in target_phrases:
    matched = False
    for line in lines:
        if line.strip().startswith(phrase):
            rects = page.search_for(line.strip())
            if rects:
                for rect in rects:
                    highlight = page.add_highlight_annot(rect)
                    highlight.update()
                matched = True
                break
    if not matched:
        print(f"❌ Could not find line starting with: {phrase}")

# Save updated PDF
doc.save(output_pdf, garbage=4, deflate=True)
doc.close()

print(f"✅ Highlighted PDF saved to: {output_pdf}")
