import torch
import torch.nn as nn
from torch.autograd import Variable
from data_loading import *
from rdkit import Chem

'''
the model
'''
class generative_model(nn.Module):
    def __init__(self, vocabs_size, hidden_size, output_size, embedding_dimension, n_layers):
        super(generative_model, self).__init__()
        self.vocabs_size = vocabs_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_dimension = embedding_dimension
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocabs_size, embedding_dimension)
        self.rnn = nn.LSTM(embedding_dimension, hidden_size, n_layers, dropout = 0.2)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        batch_size = input.size(0)
        input = self.embedding(input)
        output, hidden = self.rnn(input.view(1, batch_size, -1), hidden)
        output = self.linear(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        hidden=(Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return hidden

file=open('results/generated_smiles_non-redundant.txt','a')
data,vocabs=load_data()
data = set(list(data))
vocabs = list(vocabs)
vocabs_size = len(vocabs)
output_size = len(vocabs)
batch_size = 128
cuda = True
hidden_size = 1024
embedding_dimension =  248
n_layers=3
end_token = ' '

model = generative_model(vocabs_size,hidden_size,output_size,embedding_dimension,n_layers)
model.load_state_dict(torch.load('mytraining.pt', map_location=torch.device('cpu')))
if cuda:
    model = model.cuda()
model.eval()


N_samples = 1000000 ## to sample 
N_batch = 1000 ##sample N_batch sequence at a time 

def evaluate(prime_str=['!'], temperature=0.4):
    max_length = 200

    ### Make the inp shape to be 2 x 1 (batch num x data)
    if cuda:
        inputs = torch.tensor([tensor_from_chars_list(s,vocabs,cuda) for s in prime_str])
        inp = Variable(inputs).cuda()
    else: 
        inputs = torch.tensor([tensor_from_chars_list(s,vocabs,cuda) for s in prime_str])
        inp = Variable(inputs)
    inp = torch.unsqueeze(inp, 1)
    batch_size = inp.size(0)
    tracker=np.array([-1 for i in range(batch_size)])  ### track when to get the str
    hidden = model.init_hidden(batch_size)

    if cuda:
        hidden = (hidden[0].cuda(), hidden[1].cuda())
    predicted = prime_str
    cur_pos=0 ### count the current position
    cur_end_token=0 ### count for how many chars has reached the end token  
    while True:
        cur_pos+=1  
        output, hidden = model(inp, hidden)
        output_dist = output.data.div(temperature).exp()
        top_is = torch.squeeze(torch.multinomial(output_dist, 1),1)
        predicted_chars = [vocabs[top_i] for top_i in top_is]
        for i in range(batch_size):
            if tracker[i]==-1:
                if predicted_chars[i]==end_token:
                    tracker[i] = cur_pos
                    cur_end_token+=1 
                elif len(predicted_chars[i])> max_length:
                    tracker[i] = -2 ## indicate max_length truncate  
        if cur_end_token== batch_size or cur_pos>max_length: ### either all data reaches end token or reach max lengths, return results 
            final_result=[]
            for idx,s in enumerate(predicted):
                if tracker[idx] !=-2:
                    s=s[1:tracker[idx]]
                else: 
                    s=s[1:]
                final_result.append(s)
            # print ('final_result',final_result)
            # sys.exit(0)
            return final_result
        predicted = [old+new for old,new in zip (predicted,predicted_chars)]
        if cuda:
            inputs = torch.tensor([tensor_from_chars_list(s,vocabs,cuda) for s in predicted_chars])
            inp = Variable(inputs).cuda()
        else: 
            inputs = torch.tensor([tensor_from_chars_list(s,vocabs,cuda) for s in predicted_chars])
            inp = Variable(inputs)
        inp = torch.unsqueeze(inp, 1)
    return predicted
def valid_smile(smile):
    return Chem.MolFromSmiles(smile) is not None
def get_canonical_smile(smile):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smile))
def valid_smiles_at_temp(temp):
    range_test = 100
    c=0
    for i in range(range_test):
        s= evaluate(prime_str='!', temperature=temp)[1:] # remove the first character !.
        if valid_smile(s):
            print(s)
            c+=1
    return float(c)/range_test
def smiles_in_db(smile):
    smile = '!'+get_canonical_smile(smile)+' '
    if smile in data:
        return True
    return False

def percentage_variety_of_valid_at_temp(temp):
    range_test = 100
    c_v=0
    c_nd=0
    for i in range(range_test):
        s= evaluate(prime_str='!', temperature=temp)[1:] # remove the first character !.
        if valid_smile(s):
            c_v+=1
            if not smiles_in_db(s):
                c_nd+=1
    return float(c_nd)/c_v

valid_smiles = []
start_token=['!' for i in range(N_batch)]
while len(valid_smiles)<N_samples:
    predicteds=evaluate(prime_str=start_token)
    for predicted in predicteds:
        if valid_smile(predicted) and not smiles_in_db(predicted):
#        if valid_smile(predicted):
            valid_smiles.append(predicted)
            file.write(predicted+'\n')
#    print ('# of valid smiles', len(valid_smiles))
file.close()
#print ('-----------------Final Results--------------------')
#valid_smiles=valid_smiles[:N_samples]
#[print (str(idx)+',',smile) for idx,smile in enumerate(valid_smiles)]
