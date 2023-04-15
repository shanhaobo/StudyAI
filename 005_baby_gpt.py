import torch
from torch.nn import functional as F
from transformer.GPTModel import GPTModel, GPTConfig

torch.manual_seed(1337)

# context length is 3, so we take 3 bits to predict the next bit probability
context_length = 3
config = GPTConfig(
    PosEmbedDim = context_length,
    # vocab size is 2, so we only have two possible tokens: 0,1
    VocabSize = 2,
    BlockNum = 4,
    HeadNum = 4,
    EmbedDim = 16,
    EnableBias = False,
)
gpt = GPTModel(config)
gpt.PrintNumParameters()
gpt.TraversePrintParameters()
gpt.TraversePrintModuleInfo()

def all_possible(n, k):
    # return all possible lists of k elements, each in range of [0,n)
    if k == 0:
        yield []
    else:
        for i in range(n):
            for c in all_possible(n, k - 1):
                yield [i] + c
list(all_possible(config.VocabSize, config.PosEmbedDim))

from graphviz import Digraph

def plot_model():
    dot = Digraph(comment='Baby GPT', engine='circo')

    for xi in all_possible(gpt.config.VocabSize, gpt.config.PosEmbedDim):
        
        # forward the GPT and get probabilities for next token
        x = torch.tensor(xi, dtype=torch.long)[None, ...] # turn the list into a torch tensor and add a batch dimension
        logits = gpt(x) # forward the gpt neural net
        probs = F.softmax(logits, dim=-1) # get the probabilities
        y = probs[0].tolist() # remove the batch dimension and unpack the tensor into simple list
        print(f"input {xi} ---> {y}")

        # also build up the transition graph for plotting later
        current_node_signature = "".join(str(d) for d in xi)
        dot.node(current_node_signature)
        for t in range(gpt.config.VocabSize):
            next_node = xi[1:] + [t] # crop the context and append the next character
            next_node_signature = "".join(str(d) for d in next_node)
            p = y[t]
            label=f"{t}({p*100:.0f}%)"
            dot.edge(current_node_signature, next_node_signature, label=label)
    
    return dot

plot_model()
