import esm
import torch
import pickle
import sys
import argparse


def get_esm_embeddings(fasta_file, model_name="esm2_t33_650M_UR50D"):
    """
    Extract ESM-2 protein representations from FASTA file.
    
    Args:
        fasta_file: Path to FASTA file with protein sequences
        model_name: ESM-2 model to use
    """
    # Load ESM-2 model and alphabet
    print(f"Loading ESM-2 model: {model_name}...")
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Parse FASTA file
    print(f"Reading sequences from {fasta_file}...")
    sequences = {}
    try:
        with open(fasta_file) as f:
            protein_id = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if protein_id and current_seq:
                        sequences[protein_id] = ''.join(current_seq)
                    protein_id = line[1:].split()[0]  # Get first part of header
                    current_seq = []
                elif line:  # Non-empty line
                    current_seq.append(line)
            # Don't forget the last sequence
            if protein_id and current_seq:
                sequences[protein_id] = ''.join(current_seq)
    except FileNotFoundError:
        print(f"Error: File '{fasta_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        sys.exit(1)
    
    if not sequences:
        print("Error: No sequences found in FASTA file")
        sys.exit(1)
    
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
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Embeddings saved to {output_file}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        sys.exit(1)


def load_embeddings(filepath="protein_embeddings.pkl"):
    """Load embeddings from pickle file."""
    try:
        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Extract ESM-2 protein representations from FASTA file'
    )
    parser.add_argument(
        '--fasta',
        type=str,
        default='string_proteins_sequences.fa',
        help='Path to FASTA file with protein sequences (default: string_proteins_sequences.fa)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='protein_embeddings.pkl',
        help='Output file for embeddings (default: protein_embeddings.pkl)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='esm2_t33_650M_UR50D',
        help='ESM-2 model to use (default: esm2_t33_650M_UR50D)'
    )
    
    args = parser.parse_args()
    
    # Extract ESM-2 protein representations
    embeddings = get_esm_embeddings(args.fasta, args.model)
    
    # Save embeddings
    save_embeddings(embeddings, args.output)
    

if __name__ == '__main__':
    main()
