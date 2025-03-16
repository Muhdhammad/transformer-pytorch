import torch
import torch.nn as nn
import math

"""
We map input token to vector representation with dimensions (d_model) and vocabulary size(vocab_size)
uses nn.Embedding to create token embeddings.
"""

class inputEmbeddings(nn.Module):

  def __init__(self, d_model, vocab_size):
    super().__init__()

    self.d_model = d_model
    self.vocab_size = vocab_size

    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    # (batch, seq_len) --> (batch, seq_len, d_model)
    # Multiply by sqrt(d_model) to scale the embeddings according to the paper
    return self.embedding(x) * math.sqrt(self.d_model)

"""
This class adds positional context to token embeddings to help attention-mechanism understand order of words.
"""

class positionalEncoding(nn.Module):
  def __init__(self, d_model, seq_len, dropout):
    super().__init__()

    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)

    # create an empty tensor to fill in of shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)
    # Numerator - we create a position vector of shape (seq_len,1)
    position = torch.arange(start=0, end=seq_len, dtype=torch.float).unsqueeze(1)
    # Division term of the formula
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    # Apply sine and cosine, sine for even indices and cosine for odd indices
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # We will have batches, so we will add new dimension in pe
    # Now, the shape will become (1, seq_len, d_model)
    pe = pe.unsqueeze(0)

    # we register the pe in the buffer of the model to make it part of the model
    # it shouldn't be updated during back prop as a trainable parameter
    # it should be constant because positional encodings don't change during training
    self.register_buffer("pe", pe)

  def forward(self, x):
    # We add the positional encoding to the input tensor.
    # We slice it to match the dimensions of the word embedding.
    # Remember, the dimensions of `self.pe` are: [1, seq_len, d_model].
    # The dimensions of `x` are: [batch_size, seq_len, d_model].
    # We take dim 1 (the sequence length of positional encodings) and align it with dim 1 of `x`
    # We use x.shape[1] to ensure that we take only as many positional encodings as there are words (tokens) in the input sequence.
    x = x + self.pe[:, :x.shape[1], :]
    return self.dropout(x)

"""
We add epsilon (eps) in Layer Normalization for numerical stability which prevents division by zero when computing the standard deviation.
"""

class layerNorm(nn.Module):
  def __init__(self, eps:float = 10**-6):
    super().__init__()
    self.eps = eps
    # we set alpha and bias as trainable parameters
    self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative
    self.bias = nn.Parameter(torch.zeros(1)) # Additive

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)

    # Now our formula
    return self.alpha * (x - mean) / (std + self.eps) + self.bias

"""
A simple feedforward layer which is a fully connected network.
It consists of two linear layers with ReLU activation and dropout in between.
"""

class feedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout):
    super().__init__()
    # Two fully connected layers with relu in between
    self.linear1 = nn.Linear(d_model, d_ff) # This is W1 and B1
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(d_ff, d_model) # this is W2 and B2

  # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
  def forward(self,x):
    x = self.relu(self.linear1(x))
    x = self.dropout(x)
    x = self.linear2(x)
    return x

"""
This is the heart of transformers: Multi-head attention
"""

class multiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads, dropout):
    super().__init__()

    self.d_model = d_model
    self.num_heads = num_heads

    if d_model % num_heads != 0:
      print("d_model is not divisible by num_heads")
    self.d_k = d_model // num_heads

    self.w_q = nn.Linear(d_model, d_model) # Wq
    self.w_k = nn.Linear(d_model, d_model) # Wk
    self.w_v = nn.Linear(d_model, d_model) # Wv

    # Set Wo, which is [(num_heads * self.d_v), d_model] = [d_model, d_model]
    self.w_o = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
    # We get d_k which is the last dimension of query, key, value
    d_k = query.shape[-1]
    # (batch, num_heads, seq_len, d_k) -> (batch, num_heads, seq_len, seq_len)
    # The result is an attention  matrix which tells how much each word attends to other
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
      attention_scores.masked_fill_(mask == 0, -1e9)

    attention_scores = attention_scores.softmax(dim = -1)
    if dropout is not None:
      attention_scores = dropout(attention_scores)

    # We return a tuple with the attention and the self-attention for visualization
    return (attention_scores @ value), attention_scores


  def forward(self, q, k, v, mask):
    query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
    key = self.w_k(k)   # same
    value = self.w_v(v) # same

    # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
    key = key.view(key.shape[0], k.shape[1], self.num_heads, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

    x, self.attention_scores = multiHeadAttention.attention(query, key, value, mask, self.dropout)

    # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k)

    # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
    return self.w_o(x)

"""
residual connection block allow model to retain original information while learning new representations
this allow faster and stable training
"""

class residualConnection(nn.Module):
  def __init__(self, dropout):
    super().__init__()

    self.dropout = nn.Dropout(dropout)
    self.norm = layerNorm()

  # Normalize x, then pass it through a sublayer(multihead or feedforward) then apply dropout
  def forward(self, x, sublayer):
    return x + self.dropout(sublayer(self.norm(x)))

"""
this is encoder block which will be used to create our main encoder class
"""

class EncoderBlock(nn.Module):
  def __init__(self, self_attention_block: multiHeadAttention, feed_forward_block: feedForward, dropout):
    super().__init__()

    self.self_attention_block = self_attention_block
    self.feed_forward_block = feed_forward_block
    # Store two residual connections in nn.ModuleList()
    self.residual_connection = nn.ModuleList([residualConnection(dropout) for _ in range(2)])

  def forward(self, x, src_mask):
    # lambda x: self.self_attention_block(x, x, x), it delay the execution and only compute it when called

    # In the first residual connection (idx: 0), we apply MultiHeadAttention, which takes Q, K, and V and mask.
    # The output of the attention block is added to the original input (residual connection).
    # This result is then passed to the second residual connection (idx: 1), which applies the FeedForward block.
    # The FeedForward output is again added to its input (residual connection) before returning the final output.

    x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
    x = self.residual_connection[1](x, self.feed_forward_block)
    return x

"""
This is our main Encoder module, which stacks multiple Encoder blocks.
Since the Encoder consists of several layers, we use nn.ModuleList()
Each layer applies self-attention and feedforward transformations with residual connections
Finally, a LayerNorm is applied before returning the output.
"""

class Encoder(nn.Module):
  def __init__(self, layers: nn.ModuleList):
    super().__init__()

    self.layers = layers
    self.norm = layerNorm()

  def forward(self, x, mask):
    # Pass input through each encoder layer sequentially
    for layer in self.layers:
      x = layer(x, mask)

    # Apply final layer normalization
    return self.norm(x)

"""
this is decoder block which will be used to create our main decoder class
"""

class DecoderBlock(nn.Module):
  def __init__(self, self_attention_block: multiHeadAttention,
               cross_attention_block: multiHeadAttention,
               feed_forward_block: feedForward,
               dropout):
    super().__init__()

    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.feed_forward_block = feed_forward_block

    # we have three residual connections
    self.residual_connections = nn.ModuleList([residualConnection(dropout) for _ in range(3)])

  def forward(self, x, encoder_output, src_mask, target_mask):

    # First residual connection: Applies self-attention on the decoder input (masked multi-head self-attention).
    # target_mask to prevent attending to future tokens
    x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))

    # Second residual connection: Applies cross-attention between the decoder input and encoder outputs.
    # The decoder input acts as the query, while the encoder outputs serve as the key and value.
    x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
    x = self.residual_connections[2](x, self.feed_forward_block)
    return x

"""
This is our main Decoder module, which stacks multiple Decoder blocks.
Like the Encoder, it consists of multiple layers stored in nn.ModuleList().
Each layer applies self-attention, cross-attention with the encoder output, and feedforward transformations, all with residual connections.
Finally, a LayerNorm is applied before returning the output.
"""

class Decoder(nn.Module):
  def __init__(self, layers: nn.ModuleList):
    super().__init__()

    self.layers = layers
    self.norm = layerNorm()

  def forward(self, x, encoder_output, src_mask, target_mask):
    for layer in self.layers:
      x = layer(x, encoder_output, src_mask, target_mask)
    return self.norm(x)

"""
The last linear layer in transformer serves as projection layer that maps the final hidden state to output vocab logits.
decoder gives hidden state of shape (batch_size, seq_len, d_model)
last linear layer transforms this into (batch_size, seq_len, vocab_size)
this allow us to apply softmax over vocab_size, giving probability for each token in vocabulary
"""

class LinearLayer(nn.Module):
  def __init__(self, d_model, vocab_size):
    super().__init__()

    self.linear_proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    # (batch_size, seq_len, d_model) -> # (batch_size, seq_len, vocab_size)
    return torch.log_softmax(self.linear_proj(x), dim = -1)

"""
This is the main class: Transformer, we combine all the blocks in this class
"""

class Transformer(nn.Module):
  def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: inputEmbeddings, target_embedding: inputEmbeddings, src_pos: positionalEncoding, target_pos: positionalEncoding, linear_layer: LinearLayer):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.src_embedding = src_embedding
    self.target_embedding = target_embedding
    self.src_pos = src_pos
    self.target_pos = target_pos
    self.linear_layer = linear_layer

  def encode(self, src, src_mask):
    src = self.src_embedding(src)
    src = self.src_pos(src)
    return self.encoder(src, src_mask)

  def decode(self, encoder_output, target, src_mask, target_mask):
    target = self.target_embedding(target)
    target = self.target_pos(target)
    return self.decoder(target, encoder_output, src_mask, target_mask)

  def projection(self, x):
    return self.linear_layer(x)

"""
This method constructs a complete Transformer model by assembling all components.
It includes input embeddings, positional encodings, and `n` encoder and decoder blocks.
Each encoder block consists of self-attention, cross-attention(if decoder) and feedforward layers.
Finally, a linear projection layer.
"""


def make_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int,
                      d_model: int = 512, n: int = 6, num_heads: int = 8, d_ff: int = 2048,
                      dropout: float = 0.1
                    ):

    # Create input embeddings of src and target
    src_embedding = inputEmbeddings(d_model, src_vocab_size)
    target_embedding = inputEmbeddings(d_model, target_vocab_size)

    # Create positional encoding of src and target
    src_pos = positionalEncoding(d_model, src_seq_len, dropout)
    target_pos = positionalEncoding(d_model, target_seq_len, dropout)

    # Create encoder blocks
    encoder_blocks = []

    for i in range(n):
      encoder_self_attention = multiHeadAttention(d_model, num_heads, dropout)
      feed_forward = feedForward(d_model, d_ff, dropout)
      encoder_block = EncoderBlock(encoder_self_attention, feed_forward, dropout)
      encoder_blocks.append(encoder_block)

    # Create decoder blocks
    decoder_blocks = []

    for i in range(n):
      decoder_self_attention = multiHeadAttention(d_model, num_heads, dropout)
      decoder_cross_attention = multiHeadAttention(d_model, num_heads, dropout)
      feed_forward = feedForward(d_model, d_ff, dropout)
      decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward, dropout)
      decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create linear layer (projection layer)
    linear_layer = LinearLayer(d_model, target_vocab_size)

    # Finally, create our transformer
    transformer = Transformer(encoder, decoder, src_embedding, target_embedding, src_pos, target_pos, linear_layer)

    # Initialize the parameters with Xavier initialization
    # nn.init.xavier_uniform_ initializes weights using uniform distribution
    # instead of just random weights to stabilize the training

    for p in transformer.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)

    return transformer