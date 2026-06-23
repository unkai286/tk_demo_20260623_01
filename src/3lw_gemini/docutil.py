from spire.doc import CharacterFormat, Document, FileFormat, TableRow, TextRange


def save_as_docx(inputFileName, outputFileName):
    doc = Document()
    doc.LoadFromFile(inputFileName)

    # Create a characterFormat object
    characterFormat = CharacterFormat(doc)
    # Set font
    characterFormat.FontName = "IBM Plex Sans JP Text"
    characterFormat.FontSize = 8

    # Loop through all sections and paragraphs
    for s in range(doc.Sections.Count):
        section = doc.Sections[s]

        for p in range(section.Paragraphs.Count):
            paragraph = section.Paragraphs[p]
            # Loop through the childObjects of paragraph
            for i in range(paragraph.ChildObjects.Count):
                childObj = paragraph.ChildObjects.get_Item(i)
                if isinstance(childObj, TextRange):
                    # Apply character format
                    tr = childObj if isinstance(childObj, TextRange) else None
                    tr.ApplyCharacterFormat(characterFormat)

        for t in range(section.Tables.Count):
            table = section.Tables[t]
            # Loop through the childObjects of table
            for i in range(table.ChildObjects.Count):
                # row
                childObj = table.ChildObjects.get_Item(i)
                if isinstance(childObj, TableRow):
                    # cell
                    for c in range(childObj.Cells.Count):
                        cell = childObj.Cells[c]
                        for p in range(cell.Paragraphs.Count):
                            para = cell.Paragraphs[p]
                            for j in range(para.ChildObjects.Count):
                                cellChildObj = para.ChildObjects.get_Item(j)
                                if isinstance(cellChildObj, TextRange):
                                    cellChildObj.ApplyCharacterFormat(characterFormat)

    doc.SaveToFile(outputFileName, FileFormat.Docx)
    doc.Close()
