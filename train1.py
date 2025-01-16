import os
import urllib.request
import ssl
import torch
import torch.nn as nn
from collections import Counter
from torchtext.vocab import vocab
import spacy
from model import Seq2SeqTransformer
from tqdm import tqdm
from datasets import load_dataset
from torch.cuda.amp import GradScaler, autocast
import warnings
warnings.filterwarnings('ignore')

print(torch.cuda.is_available())  # Should return True if GPU is accessible
print(torch.cuda.get_device_name(0))  # Displays the name of the GPU

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def download_dataset():
    try:
        print("Downloading dataset from Hugging Face...")
        dataset = load_dataset("opus_books", "de-en", split='train')
        
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Filter criteria
        def is_valid_sentence(text):
            # Skip metadata, titles, and very short texts
            skip_starts = ['Source:', 'Chapter', 'Kapitel', 'http', 'www']
            return (len(text.split()) >= 3 and  # At least 3 words
                    not any(text.startswith(s) for s in skip_starts) and
                    not text.isupper() and  # Skip all-caps titles
                    '.' in text)  # Likely a real sentence
        
        # Write to files, only including valid sentences
        with open('data/train.de', 'w', encoding='utf-8') as f_de, \
             open('data/train.en', 'w', encoding='utf-8') as f_en:
            for example in dataset:
                de_text = example['translation']['de']
                en_text = example['translation']['en']
                if is_valid_sentence(de_text) and is_valid_sentence(en_text):
                    f_de.write(de_text + '\n')
                    f_en.write(en_text + '\n')
        
        print("Dataset downloaded and filtered successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return False
    
def display_dataset_examples(num_examples=5):
    """
    Display example translations from the filtered dataset.
    """
    try:
        # Read directly from the filtered files
        with open('data/train.de', 'r', encoding='utf-8') as f_de, \
             open('data/train.en', 'r', encoding='utf-8') as f_en:
            de_lines = f_de.readlines()
            en_lines = f_en.readlines()
            
        print(f"\n{'-'*50}")
        print("Filtered Dataset Examples:")
        print(f"{'-'*50}")
        
        # Get random indices for examples
        import random
        total_examples = len(de_lines)
        indices = random.sample(range(total_examples), min(num_examples, total_examples))
        
        for i, idx in enumerate(indices):
            print(f"\nExample {i+1}:")
            print(f"German    : {de_lines[idx].strip()}")
            print(f"English   : {en_lines[idx].strip()}")
        
        print(f"\n{'-'*50}")
        print(f"Total filtered sentences: {total_examples}")
        
    except Exception as e:
        print(f"Error accessing dataset: {str(e)}")

# Load the data
def load_data(src_path, tgt_path):
    with open(src_path, 'r', encoding='utf-8') as f:
        src_sentences = f.readlines()
    with open(tgt_path, 'r', encoding='utf-8') as f:
        tgt_sentences = f.readlines()
    return list(zip(src_sentences, tgt_sentences))

# Load spaCy models
try:
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy models...")
    os.system('python -m spacy download de_core_news_sm')
    os.system('python -m spacy download en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

# Special tokens
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Language settings
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Tokenization functions
def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text.strip())]

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text.strip())]

token_transform = {}
token_transform[SRC_LANGUAGE] = tokenize_de
token_transform[TGT_LANGUAGE] = tokenize_en

# Build vocab function
def build_vocab(data_iter, tokenizer):
    counter = Counter()
    for text, _ in data_iter:
        counter.update(tokenizer(text))
    vocab_obj = vocab(counter, min_freq=2, specials=special_symbols, special_first=True)

    # Set the default index for unknown tokens to UNK_IDX
    vocab_obj.set_default_index(UNK_IDX)
    
    return vocab_obj

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_tokens = token_transform[SRC_LANGUAGE](src_sample)
        tgt_tokens = token_transform[TGT_LANGUAGE](tgt_sample)
        
        src_tokens = [BOS_IDX] + [vocab_transform[SRC_LANGUAGE][token] for token in src_tokens] + [EOS_IDX]
        tgt_tokens = [BOS_IDX] + [vocab_transform[TGT_LANGUAGE][token] for token in tgt_tokens] + [EOS_IDX]
        
        src_batch.append(torch.tensor(src_tokens))
        tgt_batch.append(torch.tensor(tgt_tokens))
    
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask.to(DEVICE), tgt_mask.to(DEVICE), src_padding_mask.to(DEVICE), tgt_padding_mask.to(DEVICE)

def tokenize_pair(src_text, tgt_text):
    """Tokenize a pair of sentences and check token coverage"""
    src_tokens = tokenize_de(src_text)
    tgt_tokens = tokenize_en(tgt_text)
    
    # Check source token coverage
    src_in_vocab = [token for token in src_tokens if token in vocab_transform[SRC_LANGUAGE]]
    src_coverage = len(src_in_vocab) / len(src_tokens) if src_tokens else 0
    
    # Check target token coverage
    tgt_in_vocab = [token for token in tgt_tokens if token in vocab_transform[TGT_LANGUAGE]]
    tgt_coverage = len(tgt_in_vocab) / len(tgt_tokens) if tgt_tokens else 0
    
    return src_tokens, tgt_tokens, src_coverage, tgt_coverage

def validate_batch(model, val_dataloader, loss_fn, num_batches=5):
    """Validate the model on a few batches and show example translations"""
    model.eval()
    total_loss = 0
    example_count = 0
    
    print("\nValidation Examples:")
    print("-" * 50)
    
    with torch.no_grad():
        for idx, (src, tgt) in enumerate(val_dataloader):
            if idx >= num_batches:
                break
                
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
            
            logits = model(src, tgt_input, src_mask, tgt_mask,
                          src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()
            
            # Show example translations
            if example_count < 3:  # Show first 3 examples from batch
                for b in range(min(3, src.size(1))):
                    src_tokens = [vocab_transform[SRC_LANGUAGE].get_itos()[idx] for idx in src[:, b] if idx not in [PAD_IDX, EOS_IDX]]
                    tgt_tokens = [vocab_transform[TGT_LANGUAGE].get_itos()[idx] for idx in tgt[1:, b] if idx not in [PAD_IDX, EOS_IDX]]
                    print(f"\nExample {example_count + 1}:")
                    print(f"Source: {' '.join(src_tokens)}")
                    print(f"Target: {' '.join(tgt_tokens)}")
                    example_count += 1
                    if example_count >= 3:
                        break
    
    return total_loss / num_batches

def train_epoch(model, optimizer, train_dataloader, loss_fn):
    model.train()
    losses = 0
    train_iter = tqdm(train_dataloader, desc="Training")
    
    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    for src, tgt in train_iter:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        
        optimizer.zero_grad()

        # Forward pass with autocast for mixed precision
        with autocast():
            logits = model(src, tgt_input, src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        # Backward pass with scaled gradients
        scaler.scale(loss).backward()

        # Step the optimizer and update the scaler
        scaler.step(optimizer)
        scaler.update()

        losses += loss.item()
        
    return losses / len(train_dataloader)

def main():
    print(f"Using device: {DEVICE}")
    
    # Download the dataset
    if not download_dataset():
        print("Error downloading dataset. Please check the URLs or download manually.")
        return
    
    # Display some example translations
    print("\nShowing sample translations from the dataset...")
    display_dataset_examples()
    
    # Load the training data
    print("Loading training data...")
    train_data = load_data('data/train.de', 'data/train.en')
    print(f"Loaded {len(train_data)} training pairs")
    
    # Create vocabularies
    print("Building vocabularies...")
    global vocab_transform
    vocab_transform = {}
    vocab_transform[SRC_LANGUAGE] = build_vocab(train_data, tokenize_de)
    vocab_transform[TGT_LANGUAGE] = build_vocab(train_data, tokenize_en)
    
    # Create DataLoader
    BATCH_SIZE = 32
    train_dataloader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Model parameters
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    
    print("Initializing model...")
    # Initialize model
    transformer = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
        NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM
    ).to(DEVICE)
    
    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    # Training
    print("Starting training...")
    NUM_EPOCHS = 20
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(transformer, optimizer, train_dataloader, loss_fn)
        print(f"Epoch: {epoch+1}, Train loss: {train_loss:.3f}")
        torch.cuda.empty_cache()
        # Save checkpoint after each epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pt')
    
    # Save the final model and vocabularies
    print("Saving model and vocabularies...")
    torch.save(transformer.state_dict(), 'transformer_de_to_en_model.pt')
    torch.save(vocab_transform[SRC_LANGUAGE], 'vocab_de.pth')
    torch.save(vocab_transform[TGT_LANGUAGE], 'vocab_en.pth')
    print("Training completed!")

if __name__ == "__main__":
    main()