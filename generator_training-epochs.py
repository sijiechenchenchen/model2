import torch
import torch.nn as nn
from torch.autograd import Variable
from data_loading import *
from torch.utils.tensorboard import SummaryWriter 
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

data,vocabs=load_data()
data=data[:1280]  ### smaller dataset to debug your code  
vocabs = list(vocabs)
vocabs_size = len(vocabs)
output_size = len(vocabs)
batch_size = 20
cuda = False
hidden_size = 1024
embedding_dimension =  248
n_layers=3
lr = 0.0001 #lr = 0.005
n_batches=5
#n_batches =  200000
#print_every = 100
print_every = 100
plot_every = 10
#save_every = 100
save_every = 1#1000
end_token = ' '

print(len(data))
print("processing batches ...")
train_batches,val_batches = process_data_to_batches(data,batch_size,vocabs,cuda)
#print(len(train_batches))
print("finish processing batches")


model = generative_model(vocabs_size,hidden_size,output_size,embedding_dimension,n_layers)
#model.load_state_dict(torch.load('mytraining.pt', map_location=torch.device('cpu')))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

def propagation(inp, target, mode):
    batch_size = inp.size(0)
    sequence_length = inp.size(1)
    hidden = model.init_hidden(batch_size)
    if cuda:
        hidden = (hidden[0].cuda(), hidden[1].cuda())
    if mode=='train':
        model.zero_grad()
    loss = 0
    for c in range(sequence_length):
        output, hidden = model(inp[:,c], hidden)
        loss += criterion(output, target[:,c])
    if mode=='train':
        loss.backward()
        optimizer.step()
        return loss.item()/sequence_length
    elif model=='val':
        return loss.item()/sequence_length
#    return loss.data[0]/sequence_length


def evaluate(prime_str='!', temperature=0.4):
    max_length = 200
    inp = Variable(tensor_from_chars_list(prime_str,vocabs))
    if cuda:
        inp = Variable(tensor_from_chars_list(prime_str,vocabs,cuda)).cuda()
    batch_size = inp.size(0)
    hidden = model.init_hidden(batch_size)
    if cuda:
        hidden = (hidden[0].cuda(), hidden[1].cuda())
    predicted = prime_str
    while True:
        output, hidden = model(inp, hidden)
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        # Add predicted character to string and use as next input
        predicted_char = vocabs[top_i]

        if predicted_char ==end_token or len(predicted)>max_length:
            return predicted

        predicted += predicted_char
        inp = Variable(tensor_from_chars_list(predicted_char,vocabs))
        if cuda:
            inp = Variable(tensor_from_chars_list(predicted_char,vocabs,cuda)).cuda()
    return predicted

start = time.time()
all_losses = []
loss_avg = 0

Result_folder='results/'  ### Tong: add a folder to collect results 
writer = SummaryWriter(Result_folder+'logs') ### Tong: initialize tensorboard writer 
import random       

#epochs=2000
epochs=5
#print(len(train_batches))
train_batches=list(train_batches)
for epoch in range(epochs):
    random.shuffle(train_batches)
    avg_loss=0
    total_loss=0
    for train_idx,train in enumerate(train_batches):
        print(train_idx,train)
        inp,target=train
        loss = propagation(inp, target, 'train') 
        total_loss += loss
    avg_loss=torch.mean(total_loss)
    writer.add_scalar('Training/Loss',avg_loss, epoch) 
    if epoch % print_every == 0:
       #print('[%s (%d %d%%) %.4f]' % (time_since(start), batch, batch / n_batches * 100, loss))
        print(evaluate('!'), '\n')
    if epoch %save_every ==0:
        print('[Debug] Saving model')
        torch.save(model.state_dict(), Result_folder+'models/mytraining_{}.pt'.format(batch)) ### Tong: add index to avoid model overwrite 
        print ('[Debug] Finished saving model')
    if epoch % eval_every ==0:
        with torch.no_grad():
            val_total_loss=0
            for val_idx,val_data in enumerate(val_batches):
                inp,target=val_data
                loss= propagation(inp,target,'val')
                val_total_loss+=loss
            val_avg_loss=torch.mean(val_total_loss)
            writer.add_scalar('Test/Loss',val_avg_loss, epoch) 






