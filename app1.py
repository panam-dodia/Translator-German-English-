#app.py
import streamlit as st
import torch
from model import Seq2SeqTransformer
import spacy

# Load spaCy models
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Special tokens
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_source_mask(src):
    src_seq_len = src.shape[0]
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    return src_mask

def preprocess_text(text):
    """Preprocess text consistently with training"""
    # Lowercase the text
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def debug_tokens(tokens, vocab_de):
    """Print debugging information about token processing"""
    debug_info = []
    for token in tokens:
        in_vocab = token in vocab_de
        debug_info.append(f"{token}: {'In vocabulary' if in_vocab else 'NOT in vocabulary'}")
    return debug_info

def translate(model, text, vocab_de, vocab_en):
    model.eval()
    
    # Preprocess input text
    text = preprocess_text(text)
    
    # Tokenize input text
    tokens = [tok.text for tok in spacy_de.tokenizer(text)]
    
    # Debug token information
    debug_info = debug_tokens(tokens, vocab_de)
    st.write("Token debugging information:")
    for info in debug_info:
        st.write(info)
    
    # Handle unknown tokens
    token_ids = [vocab_de.get_stoi().get(token, UNK_IDX) for token in tokens]
    
    # Add BOS and EOS tokens
    src = torch.tensor([BOS_IDX] + token_ids + [EOS_IDX], dtype=torch.long, device=DEVICE).view(-1, 1)
    src_mask = create_source_mask(src)
    
    # Generate translation
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[BOS_IDX]], dtype=torch.long, device=DEVICE)
        max_length = 100
        
        for i in range(max_length):
            memory = memory.to(DEVICE)
            tgt_mask = generate_square_subsequent_mask(ys.size(0))
            
            out = model.decode(ys, memory, tgt_mask)
            prob = model.generator(out[-1, :])
            next_word = prob.argmax().item()
            st.write(f"Step {i}: Predicted token index: {next_word}")
            
            if next_word == EOS_IDX:
                break
                
            ys = torch.cat([
                ys, 
                torch.tensor([[next_word]], dtype=torch.long, device=DEVICE)
            ], dim=0)
            
            if ys.size(0) >= max_length:
                break
    
    # Convert indices back to words
    ys = ys.cpu().numpy()
    translated_tokens = []
    for token in ys:
        if token[0] in [BOS_IDX, EOS_IDX, PAD_IDX]:
            continue
        word = vocab_en.lookup_token(token[0])
        translated_tokens.append(word)
    
    return ' '.join(translated_tokens)

def main():
    st.title("German to English Translator")
    
    try:
        # Load vocabularies
        vocab_de = torch.load('vocab_de.pth')
        vocab_en = torch.load('vocab_en.pth')
        
        # Display vocabulary sizes
        st.write(f"German vocabulary size: {len(vocab_de)}")
        st.write(f"English vocabulary size: {len(vocab_en)}")

        # Model parameters (must match training)
        SRC_VOCAB_SIZE = len(vocab_de)
        TGT_VOCAB_SIZE = len(vocab_en)
        EMB_SIZE = 512
        NHEAD = 8
        FFN_HID_DIM = 512
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3
        
        # Initialize model
        model = Seq2SeqTransformer(
            NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
            NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM
        ).to(DEVICE)
        
        st.write(f"Model architecture: {model}")
        st.write(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

        # Load trained model
        model.load_state_dict(torch.load('transformer_de_to_en_model.pt', map_location=DEVICE))
        
        # Create input text area
        text = st.text_area("Enter German text:", "Ein brauner Hund spielt im Schnee.")
        
        if st.button("Translate"):
            if text:
                translation = translate(model, text, vocab_de, vocab_en)
                st.write("English translation:")
                st.write(translation)
            else:
                st.warning("Please enter some German text to translate.")
                
    except FileNotFoundError:
        st.error("Required model files not found. Please run training first.")

if __name__ == "__main__":
    main()