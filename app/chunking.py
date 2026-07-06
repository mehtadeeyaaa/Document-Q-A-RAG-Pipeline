def fixed_size_chunk(text: str, chunk_size: int, overlap: int) -> list[str]:
    
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    
    chunks=[]
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += step

    return chunks

def recursive_chunk(text: str, chunk_size: int, overlap: int, separators: list[str] = None) -> list[str]:
    
    if not text.strip():
        return []
    
    if separators is None:
        separators = ["\n\n", ".", " "]

    pieces = text.split(separators[0])  # Split by the first separator

    chunks = []
    current_chunk = ""

    for piece in pieces:

        if len(current_chunk) + len(piece) <= chunk_size:  #
            current_chunk += piece + separators[0]

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            if len(piece) > chunk_size:
                # If the piece is too large, recursively chunk it using the next separator
                
                if len(separators) > 1:
                    sub_chunks = recursive_chunk(piece, chunk_size, overlap, separators[1:])
                    chunks.extend(sub_chunks)
                
                else:
                    # If no more separators are available, use fixed size chunking
                    chunks.extend(fixed_size_chunk(piece, chunk_size, overlap))
            
            else:
                current_chunk += piece + separators[0]
    
    if current_chunk:
                chunks.append(current_chunk.strip())

    return chunks