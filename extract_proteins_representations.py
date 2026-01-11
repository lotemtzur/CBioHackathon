import esm
import torch
import pickle
import sys


def get_esm_embeddings(fasta_file, model_name="esm2_t33_650M_UR50D"):
    """
    Extract ESM-2 protein representations from FASTA file.
    
    Args:
        fasta_file: Path to FASTA file with protein sequences
        model_name: ESM-2 model to use
    """
    # Load ESM-2 model and alphabet
    print(f"Loading ESM-2 model: {model_name}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_name)
    model.eval()
    
    # Parse FASTA file
    print(f"Reading sequences from {fasta_file}...")
    sequences = {}
    with open(fasta_file) as f:
        protein_id = None
        for line in f:
            if line.startswith('>'):
                protein_id = line[1:].strip().split()[0]  # Get first part of header
            else:
                sequences[protein_id] = line.strip()
    
    print(f"Found {len(sequences)} sequences")
    
    # Determine number of layers based on model
    if 't6' in model_name:
        num_layers = 6
    elif 't12' in model_name:
        num_layers = 12
    elif 't30' in model_name:
        num_layers = 30
    elif 't33' in model_name:
        num_layers = 33
    else:
        num_layers = 33  # Default
    
    # Extract embeddings
    embeddings = {}
    
    with torch.no_grad():
        for idx, (protein_id, seq) in enumerate(sequences.items()):
            # Tokenize and get embedding
            tokens = alphabet.encode(seq)
            tokens = torch.tensor([tokens])  # Add batch dimension
            
            # Get representations from last layer
            results = model(tokens, repr_layers=[num_layers])
            embedding = results["representations"][num_layers]  # Shape: (1, seq_len, embedding_dim)
            
            # Mean pooling across tokens
            embedding = embedding.mean(dim=1).squeeze(0)  # Shape: (embedding_dim,)
            embeddings[protein_id] = embedding
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(sequences)} sequences")
    
    embedding_dim = embeddings[list(embeddings.keys())[0]].shape[0]
    print(f"Embedding dimension: {embedding_dim}")
    return embeddings


def save_embeddings(embeddings, output_file="protein_embeddings.pkl"):
    """Save embeddings to pickle file."""
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {output_file}")


def load_embeddings(filepath="protein_embeddings.pkl"):
    """Load embeddings from pickle file."""
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def main():
    # Extract ESM-2 protein representations
    fasta_file = "string_proteins_sequences.fa"
    embeddings = get_esm_embeddings(fasta_file)
    
    # Save embeddings
    save_embeddings(embeddings)
    

if __name__ == '__main__':
    main()
